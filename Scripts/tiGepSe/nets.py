import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid

tfk = tf.keras
tfkl = tf.keras.layers


class WeakRM(tf.keras.Model):

    def __init__(self):
        super(WeakRM, self).__init__()

        self.conv1 = tfkl.Conv1D(32, 15, padding='same', activation='relu')
        self.conv2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                 kernel_regularizer=l2(0.005))
        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.25)
        self.pool1 = tfkl.MaxPool1D(pool_size=2)

        self.att_v = tfkl.Dense(128, activation='tanh')
        self.att_u = tfkl.Dense(128, activation='sigmoid')

        self.attention_weights = tfkl.Dense(1)

        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag = tf.squeeze(inputs, axis=0)

        inst_conv1 = self.conv1(input_bag)
        inst_pool1 = self.pool1(inst_conv1)
        inst_pool1 = self.dropout1(inst_pool1, training=training)

        inst_conv2 = self.conv2(inst_pool1)
        inst_conv2 = self.dropout2(inst_conv2, training=training)

        inst_features = tfkl.Flatten()(inst_conv2)

        attention_vmatrix = self.att_v(inst_features)
        attention_umatrix = self.att_u(inst_features)

        gated_attention = self.attention_weights(attention_vmatrix * attention_umatrix)

        gated_attention = tf.transpose(gated_attention, perm=[1, 0])
        gated_attention = tfkl.Softmax()(gated_attention)

        bag_features = tf.matmul(gated_attention, inst_features)

        bag_probability = self.classifier(bag_features)

        return bag_probability


class tiGepSe(tf.keras.Model):

    def __init__(self):
        super(tiGepSe, self).__init__()

        self.conv_a1 = tfkl.Conv1D(32, 15, padding='same', activation='relu')
        self.conv_a2 = tfkl.Conv1D(16, 5, padding='same', activation='relu',
                                   kernel_regularizer=l2(0.005))

        self.conv_b1 = tfkl.Conv1D(64, 5, padding='same', activation='relu')
        self.conv_b2 = tfkl.Conv1D(32, 3, padding='same', activation='relu',
                                   kernel_regularizer=l2(0.001))

        self.dropout1 = tfkl.Dropout(0.25)
        self.dropout2 = tfkl.Dropout(0.5)
        self.pool = tfkl.MaxPool1D(pool_size=2)

        self.att_v1 = tfkl.Dense(128, activation='tanh')
        self.att_u1 = tfkl.Dense(128, activation='sigmoid')

        self.att_v2 = tfkl.Dense(128, activation='tanh', kernel_regularizer=l2(0.001))
        self.att_u2 = tfkl.Dense(128, activation='sigmoid', kernel_regularizer=l2(0.001))

        self.aw1 = tfkl.Dense(1)
        self.aw2 = tfkl.Dense(1)

        self.fc1 = tfkl.Dense(256, activation='relu', kernel_regularizer=l2(0.001))
        self.fc2 = tfkl.Dense(128, activation='relu')
        self.classifier = tfkl.Dense(1, activation='sigmoid')

    def call(self, inputs, training=True, mask=None):
        input_bag, input_geo = inputs

        input_bag = tf.squeeze(input_bag, axis=0)

        h1 = self.conv_a1(input_bag)
        h1 = self.pool(h1)
        h1 = self.dropout1(h1, training=training)

        h1 = self.conv_a2(h1)
        h1 = self.dropout1(h1, training=training)
        h1 = tfkl.Flatten()(h1)

        vm1 = self.att_v1(h1)
        um1 = self.att_u1(h1)
        aw1 = self.aw1(vm1 * um1)
        aw1 = tf.transpose(aw1, perm=[1, 0])
        aw1 = tfkl.Softmax()(aw1)
        h1 = tf.matmul(aw1, h1)

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
        vm2 = self.att_v2(h3)
        um2 = self.att_u2(h3)
        aw2 = self.aw2(vm2 * um2)
        aw2 = tf.transpose(aw2, perm=[1, 0])
        aw2 = tfkl.Softmax()(aw2)
        h3 = tf.matmul(aw2, h3)

        h = self.fc2(h3)
        h = self.dropout2(h, training=training)
        out = self.classifier(h)

        return out