import tensorflow as tf
from tensorflow.keras.regularizers import l2
# from tensorflow_addons.activations import sparsemax

tfk = tf.keras
tfkl = tf.keras.layers
tfki = tf.keras.initializers


class conv_block(tf.keras.Model):

    def __init__(self, filter=64, kernel=5, pool=2):
        super(conv_block, self).__init__()
        self.conv = tfkl.Conv1D(filter, kernel, padding='same', activation='relu')
        self.pool = tfkl.MaxPool1D(pool)
        self.dropout = tfkl.Dropout(0.2)

    def call(self, inputs, training=None, mask=None):
        h = self.conv(inputs)
        h = self.pool(h)
        h = self.dropout(h, training=training)
        return h


class DeepPromise(tf.keras.Model):

    def __init__(self, num_class=1):
        super(DeepPromise, self).__init__()

        self.conv_1 = conv_block()
        self.conv_2 = conv_block()
        self.conv_3 = conv_block()
        self.conv_4 = conv_block()

        self.fc = tfkl.Dense(64, activation='relu')
        self.dropout = tfkl.Dropout(0.2)
        self.classifier = tfkl.Dense(num_class, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_seq = inputs

        h1 = self.conv_1(input_seq, training=training)
        h1 = self.conv_2(h1, training=training)
        h1 = self.conv_3(h1, training=training)
        h1 = self.conv_4(h1, training=training)
        h1 = tfkl.Flatten()(h1)

        h = self.fc(h1)
        h = self.dropout(h, training=training)
        out = self.classifier(h)
        return out


class DeepRiPe(tf.keras.Model):
    def __init__(self, num_class=1):
        super(DeepRiPe, self).__init__()

        self.conv_a1 = tfkl.Conv1D(90, 7, padding='valid', activation='relu')
        self.pool1 = tfkl.MaxPool1D(4, 2)

        self.conv_b1 = tfkl.Conv1D(90, 7, padding='valid', activation='relu')
        self.pool2 = tfkl.MaxPool1D(10, 5)

        self.conv_m = tfkl.Conv1D(100, 5, padding='valid', activation='relu')
        self.pool3 = tfkl.MaxPool1D(10, 5)

        self.fc = tfkl.Dense(250, activation='relu')
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.5)
        self.classifier = tfkl.Dense(num_class, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_seq, input_geo = inputs

        h1 = self.conv_a1(input_seq)
        h1 = self.pool1(h1)
        h1 = self.dropout1(h1, training=training)

        h2 = self.conv_b1(input_geo)
        h2 = self.pool2(h2)
        h2 = self.dropout1(h2, training=training)

        h = tf.concat([h1, h2], axis=-2)
        h = self.conv_m(h)
        h = self.pool3(h)
        h = self.dropout1(h, training=training)
        h = tfkl.Flatten()(h)

        h = self.fc(h)
        out = self.classifier(h)
        return out


class GepSe(tf.keras.Model):

    def __init__(self, num_class=1):
        super(GepSe, self).__init__()

        self.conv_1 = conv_block()
        self.conv_2 = conv_block()
        self.conv_3 = conv_block()
        self.conv_4 = conv_block()

        self.conv_a1 = tfkl.Conv1D(64, 5, padding='same', activation='relu')
        self.conv_a2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001))
        self.pool = tfkl.MaxPool1D(pool_size=2)

        self.fc = tfkl.Dense(128, activation='relu')
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.5)
        self.classifier = tfkl.Dense(num_class, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_seq, input_geo = inputs

        h1 = self.conv_1(input_seq, training=training)
        h1 = self.conv_2(h1, training=training)
        h1 = self.conv_3(h1, training=training)
        h1 = self.conv_4(h1, training=training)
        h1 = tfkl.Flatten()(h1)

        h2 = self.conv_a1(input_geo)
        h2 = self.pool(h2)
        h2 = self.dropout1(h2, training=training)
        h2 = self.conv_a2(h2)
        h2 = self.dropout1(h2, training=training)
        h2 = tfkl.Flatten()(h2)

        h = tf.concat([h1, h2], axis=1)

        h = self.fc(h)
        h = self.dropout2(h, training=training)
        out = self.classifier(h)
        return out


class iGepSe(tf.keras.Model):

    def __init__(self, num_class=1):
        super(iGepSe, self).__init__()

        self.conv_1 = conv_block()
        self.conv_2 = conv_block()
        self.conv_3 = conv_block()
        self.conv_4 = conv_block()

        self.conv_b1 = tfkl.Conv1D(64, 5, padding='same', activation='relu')
        self.conv_b2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001))

        self.att_v = tfkl.Dense(128, activation='tanh', kernel_regularizer=l2(0.001))
        self.att_u = tfkl.Dense(128, activation='sigmoid', kernel_regularizer=l2(0.001))
        self.att_w = tfkl.Dense(num_class)

        self.fc1 = tfkl.Dense(256, activation='relu', kernel_regularizer=l2(0.001))
        self.fc2 = tfkl.Dense(128, activation='relu')
        self.classifier = tfkl.Dense(num_class, activation='sigmoid')

        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.5)
        self.pool = tfkl.MaxPool1D(pool_size=2)

    def call(self, inputs, training=True, mask=None):
        input_seq, input_geo = inputs

        h1 = self.conv_1(input_seq, training=training)
        h1 = self.conv_2(h1, training=training)
        h1 = self.conv_3(h1, training=training)
        h1 = self.conv_4(h1, training=training)
        h1 = tfkl.Flatten()(h1)

        input_geo = tf.squeeze(input_geo, axis=0)

        h2 = self.conv_b1(input_geo)
        h2 = self.pool(h2)
        h2 = self.dropout1(h2, training=training)
        h2 = self.conv_b2(h2)
        h2 = self.dropout1(h2, training=training)
        h2 = tfkl.Flatten()(h2)

        h1 = tf.broadcast_to(h1, (tf.shape(h2)[0], tf.shape(h1)[1]))

        h3 = tf.concat([h1, h2], axis=-1)
        h3 = self.fc1(h3)
        h3 = self.dropout1(h3)
        att_vm = self.att_v(h3)
        att_um = self.att_u(h3)
        att_wm = self.att_w(att_vm * att_um)
        att_wm = tf.transpose(att_wm, perm=[1, 0])
        att_wm = tfkl.Softmax()(att_wm)
        h3 = tf.matmul(att_wm, h3)

        h = self.fc2(h3)
        h = self.dropout2(h)
        out = self.classifier(h)
        return out


