import time
import itertools
import numpy as np
import tensorflow as tf
from nets import DeepPromise, DeepRiPe, GepSe, iGepSe
from utils import create_folder

tfk = tf.keras
tfdd = tf.data.Dataset


def train_model(c):
    print('Loading data!')

    train_seq = []
    train_out = []
    for i in c.train_idx:
        train_seq.append(np.load(c.data_dir + 'fold' + str(i) + '_token.npy', allow_pickle=True))
        train_out.append(np.load(c.data_dir + 'fold' + str(i) + '_label.npy', allow_pickle=True))
    train_seq = np.concatenate(train_seq)
    train_out = np.concatenate(train_out).astype(np.int32).reshape(-1)

    if c.up:
        pidx = train_out == 1
        nidx = train_out == 0
        train_seq = np.concatenate([np.repeat(train_seq[pidx], c.up, axis=0), train_seq[nidx]])
        train_out = np.concatenate([np.ones(sum(pidx) * c.up), np.zeros(sum(nidx))])
    sidx = np.random.permutation(train_out.shape[0])
    train_seq = train_seq[sidx]
    train_out = train_out[sidx]

    valid_seq = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_token.npy', allow_pickle=True)
    valid_out = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_label.npy', allow_pickle=True)

    train_out = train_out.astype(np.int32).reshape(-1,1)
    valid_out = valid_out.astype(np.int32).reshape(-1,1)

    train_seq = np.eye(4)[train_seq - 1].astype(np.float32)
    valid_seq = np.eye(4)[valid_seq - 1].astype(np.float32)

    if c.tx:
        train_geo = []
        for i in c.train_idx:
            train_geo.append(np.load(c.data_dir + 'fold' + str(i) + '_' + c.geo_enc + '_' + c.tx + '.npy', allow_pickle=True))
        train_geo = np.concatenate(train_geo)
        if c.up:
            train_geo = np.concatenate([np.repeat(train_geo[pidx], c.up, axis=0), train_geo[nidx]])
        train_geo = train_geo[sidx]
        valid_geo = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_' + c.geo_enc + '_' + c.tx + '.npy', allow_pickle=True)

    if c.tx == 'all':
        train_dataset = tfdd.from_generator(lambda: itertools.zip_longest(train_seq, train_geo, train_out),
                                            output_types=(tf.float32, tf.float32, tf.int32),
                                            output_shapes=(tf.TensorShape([501, 4]),
                                                           tf.TensorShape([None, c.window, 7]), # 7 is for chunkTX
                                                           tf.TensorShape([1])))
        valid_dataset = tfdd.from_generator(lambda: itertools.zip_longest(valid_seq, valid_geo, valid_out),
                                            output_types=(tf.float32, tf.float32, tf.int32),
                                            output_shapes=(tf.TensorShape([501, 4]),
                                                           tf.TensorShape([None, c.window, 7]),
                                                           tf.TensorShape([1])))

        train_dataset = train_dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(1)
    elif c.tx == 'long':
        train_geo = train_geo.astype(np.float32)
        valid_geo = valid_geo.astype(np.float32)

        train_dataset = tfdd.from_tensor_slices((train_seq, train_geo, train_out))
        valid_dataset = tfdd.from_tensor_slices((valid_seq, valid_geo, valid_out))

        train_dataset = train_dataset.shuffle(256).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(128)
    elif not c.tx:
        train_dataset = tfdd.from_tensor_slices((train_seq, train_out))
        valid_dataset = tfdd.from_tensor_slices((valid_seq, valid_out))

        train_dataset = train_dataset.shuffle(256).batch(128).prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(128)
    else:
        raise('Currently only all TXs (all), longest TXs (long) and no TXs (None) are available.')

    print('Creating model!')
    if isinstance(c.model_name, str):
        if not c.tx:
            dispatcher = {'DeepPromise': DeepPromise}
        elif c.tx == 'all':
            dispatcher = {'iGepSe': iGepSe}
        elif c.tx == 'long':
            dispatcher = {'GepSe': GepSe,
                          'DeepRiPe': DeepRiPe}
        try:
            model_funname = dispatcher[c.model_name]
        except KeyError:
            raise ValueError('Invalid model name')

    model = model_funname()

    opt = tfk.optimizers.Adam(lr=c.lr_init, decay=c.lr_decay)

    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_auroc = tf.keras.metrics.AUC()
    valid_auroc = tf.keras.metrics.AUC()
    train_auprc = tf.keras.metrics.AUC(curve='PR')
    valid_auprc = tf.keras.metrics.AUC(curve='PR')

    if c.tx == 'all':
        train_step_signature = [
            tf.TensorSpec(shape=(1, 501, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(1, None, c.window, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(1, 1), dtype=tf.int32)
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(train_seq, train_geo, train_out):
            with tf.GradientTape() as tape:
                prob = model((train_seq, train_geo), training=True)
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
                total_loss = loss + tf.reduce_sum(model.losses)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                opt.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_auroc(y_true=train_out, y_pred=prob)
                train_auprc(y_true=train_out, y_pred=prob)

        @tf.function(input_signature=train_step_signature)
        def valid_step(valid_seq, valid_geo, valid_out):
            prob = model((valid_seq, valid_geo), training=False)
            vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

            valid_loss(vloss)
            valid_auroc(y_true=valid_out, y_pred=prob)
            valid_auprc(y_true=valid_out, y_pred=prob)
    elif c.tx == 'long':
        @tf.function()
        def train_step(train_seq, train_geo, train_out):
            with tf.GradientTape() as tape:
                prob = model((train_seq, train_geo), training=True)
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
                total_loss = loss + tf.reduce_sum(model.losses)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                opt.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_auroc(y_true=train_out, y_pred=prob)
                train_auprc(y_true=train_out, y_pred=prob)

        @tf.function()
        def valid_step(valid_seq, valid_geo, valid_out):
            prob = model((valid_seq, valid_geo), training=False)
            vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

            valid_loss(vloss)
            valid_auroc(y_true=valid_out, y_pred=prob)
            valid_auprc(y_true=valid_out, y_pred=prob)
    else:
        @tf.function()
        def train_step(train_seq, train_out):
            with tf.GradientTape() as tape:
                prob = model(train_seq, training=True)
                loss = tfk.losses.BinaryCrossentropy(from_logits=False)(y_true=train_out, y_pred=prob)
                total_loss = loss + tf.reduce_sum(model.losses)
                gradients = tape.gradient(total_loss, model.trainable_variables)
                opt.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_auroc(y_true=train_out, y_pred=prob)
                train_auprc(y_true=train_out, y_pred=prob)

        @tf.function()
        def valid_step(valid_seq, valid_out):
            prob = model(valid_seq, training=False)
            vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

            valid_loss(vloss)
            valid_auroc(y_true=valid_out, y_pred=prob)
            valid_auprc(y_true=valid_out, y_pred=prob)

    EPOCHS = c.epoch
    current_monitor = np.inf
    patient_count = 0

    for epoch in tf.range(1, EPOCHS + 1):
        train_loss.reset_states()
        valid_loss.reset_states()
        train_auroc.reset_states()
        valid_auroc.reset_states()
        train_auprc.reset_states()
        valid_auprc.reset_states()

        estime = time.time()
        if (c.tx == 'all') | (c.tx == 'long'):
            for tdata in train_dataset:
                train_step(tdata[0], tdata[1], tdata[2])
        else:
            for tdata in train_dataset:
                train_step(tdata[0], tdata[1])
        print(f'Training of epoch {epoch} finished! Time cost is {round(time.time() - estime, 2)}s')

        vstime = time.time()
        if (c.tx == 'all') | (c.tx == 'long'):
            for vdata in valid_dataset:
                valid_step(vdata[0], vdata[1], vdata[2])
        else:
            for vdata in valid_dataset:
                valid_step(vdata[0], vdata[1])

        new_valid_monitor = np.round(valid_loss.result().numpy(), 4)
        if new_valid_monitor < current_monitor:
            if c.cp_path:
                model.save_weights(c.cp_path)
                print('val_loss improved from {} to {}, saving model to {}'.
                      format(str(current_monitor), str(new_valid_monitor), c.cp_path))
            else:
                print('val_loss improved from {} to {}, saving closed'.
                      format(str(current_monitor), str(new_valid_monitor)))

            current_monitor = new_valid_monitor
            patient_count = 0
        else:
            print('val_loss did not improved from {}'.format(str(current_monitor)))
            patient_count += 1

        if patient_count == 5:
            break

        template = "Epoch {}, Time Cost: {}s, TL: {}, TAUC: {}, TAP: {}, VL:{}, VAUC: {}, VAP: {}"
        print(template.format(epoch, str(round(time.time() - vstime, 2)),
                              str(np.round(train_loss.result().numpy(), 4)),
                              str(np.round(train_auroc.result().numpy(), 4)),
                              str(np.round(train_auprc.result().numpy(), 4)),
                              str(np.round(valid_loss.result().numpy(), 4)),
                              str(np.round(valid_auroc.result().numpy(), 4)),
                              str(np.round(valid_auprc.result().numpy(), 4)),
                              )
              )