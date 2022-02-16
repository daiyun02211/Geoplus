import time
import itertools
import numpy as np
import tensorflow as tf
from nets import tiGepSe
from utils import create_folder

tfk = tf.keras
tfdd = tf.data.Dataset


def train_model(c):
    print('Loading data!')

    train_seq = []
    train_geo = []
    train_out = []
    for i in c.train_idx:
        train_seq.append(np.load(c.data_dir + 'fold' + str(i) + '_bag.npy', allow_pickle=True))
        train_geo.append(np.load(c.data_dir + 'fold' + str(i) + '_geo.npy', allow_pickle=True))
        train_out.append(np.load(c.data_dir + 'fold' + str(i) + '_label.npy', allow_pickle=True))
    train_seq = np.concatenate(train_seq)
    train_geo = np.concatenate(train_geo)
    train_out = np.concatenate(train_out)

    valid_seq = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_bag.npy', allow_pickle=True)
    valid_geo = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_geo.npy', allow_pickle=True)
    valid_out = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_label.npy', allow_pickle=True)

    train_out = train_out.astype(np.int32).reshape(-1, 1)
    valid_out = valid_out.astype(np.int32).reshape(-1, 1)

    train_dataset = tfdd.from_generator(lambda: itertools.zip_longest(train_seq, train_geo, train_out),
                                        output_types=(tf.float32, tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, c.len, 4]),
                                                       tf.TensorShape([None, 35, 7]),
                                                       tf.TensorShape([1])))
    valid_dataset = tfdd.from_generator(lambda: itertools.zip_longest(valid_seq, valid_geo, valid_out),
                                        output_types=(tf.float32, tf.float32, tf.int32),
                                        output_shapes=(tf.TensorShape([None, c.len, 4]),
                                                       tf.TensorShape([None, 35, 7]),
                                                       tf.TensorShape([1])))

    train_dataset = train_dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(1)

    print('Creating model!')
    model = tiGepSe()

    opt = tfk.optimizers.Adam(lr=c.lr_init, decay=c.lr_decay)

    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()
    train_auroc = tf.keras.metrics.AUC()
    valid_auroc = tf.keras.metrics.AUC()

    train_step_signature = [
        tf.TensorSpec(shape=(1, None, c.len, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(1, None, 35, 7), dtype=tf.float32),
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

    @tf.function(input_signature=train_step_signature)
    def valid_step(valid_seq, valid_geo, valid_out):
        prob = model((valid_seq, valid_geo), training=False)
        vloss = tfk.losses.BinaryCrossentropy()(y_true=valid_out, y_pred=prob)

        valid_loss(vloss)
        valid_auroc(y_true=valid_out, y_pred=prob)

    EPOCHS = c.epoch
    current_monitor = np.inf
    patient_count = 0

    for epoch in tf.range(1, EPOCHS + 1):
        train_loss.reset_states()
        valid_loss.reset_states()
        train_auroc.reset_states()
        valid_auroc.reset_states()

        estime = time.time()
        for tdata in train_dataset:
            train_step(tdata[0], tdata[1], tdata[2])
        print(f'Training of epoch {epoch} finished! Time cost is {round(time.time() - estime, 2)}s')

        vstime = time.time()
        for vdata in valid_dataset:
            valid_step(vdata[0], vdata[1], vdata[2])

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

        template = "Epoch {}, Time Cost: {}s, TL: {}, TAUC: {}, VL:{}, VAUC: {}"
        print(template.format(epoch, str(round(time.time() - vstime, 2)),
                              str(np.round(train_loss.result().numpy(), 4)),
                              str(np.round(train_auroc.result().numpy(), 4)),
                              str(np.round(valid_loss.result().numpy(), 4)),
                              str(np.round(valid_auroc.result().numpy(), 4)),
                              )
              )