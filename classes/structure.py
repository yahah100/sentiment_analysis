import tensorflow as tf


class Structure:

    def __init__(self, net_name):
        """
        init
        setzt die Netzwerk Struktur durch den :param net_name fest, welches dann später
        über get_main_net geholt wird
        :param net_name: String Netz Name
        """
        self.net = net_name

    def get_main_net(self, input_tensor, word_vec_embed, just_train_last_layers,
                     batch_size=100, max_seq_len=200, num_dimensions=50, reuse=False):
        """
        gibt die passende Netzwerk Struktur zurück
        :param input_tensor: Input des Netzwerks (X_train)
        :param word_vec_embed:  Embedding Matrix
        :param just_train_last_layers: placeholder, bool falls nur der letzte Layer trainiert werden soll
        :param batch_size: int Batchsize
        :param max_seq_len: int Maximale länge der einzelnen Vektoren vom Input
        :param num_dimensions: int Anzahl an Dimensionen des Embeddings
        :param reuse: bool, falls das Netzwerk zum Testen auch benutzt werden soll
        :return: Output des letzten Layers für die Logits im TFModel
        """
        if self.net == "CNN":
            return self.get_CNN(input_tensor, word_vec_embed, just_train_last_layers,
                                batch_size, max_seq_len, num_dimensions, reuse)
        elif self.net == "LSTM":
            return self.get_LSTM(input_tensor, word_vec_embed, just_train_last_layers,
                                 batch_size, max_seq_len, num_dimensions, reuse)
        elif self.net == "ONLY_EMBED":
            return self.get_only_embed(input_tensor, word_vec_embed,
                                       batch_size, max_seq_len, num_dimensions, reuse)
        elif self.net == "WITHOUT_EMBED":
            return self.get_without_embed(input_tensor, just_train_last_layers, reuse)

    def get_CNN(self, input_tensor, word_vec_embed, just_train_last_layers,
                batch_size, max_seq_len, num_dimensions, reuse):
        """
        gibt die CNN Struktur zurück
        mit Embedding Lookup + 6 mal (Conv + Pooling) + Dropout + Output Dense
        conv1d wurde verwendet, da der Output des Lookups 3 Dimensionen hat und conv2d keine bessere Accuracy erreichte
        :param input_tensor: Input des Netzwerks (X_train)
        :param word_vec_embed:  Embedding Matrix
        :param just_train_last_layers: placeholder, bool falls nur der letzte Layer trainiert werden soll
        :param batch_size: int Batchsize
        :param max_seq_len: int Maximale länge der einzelnen Vektoren vom Input
        :param num_dimensions: int Anzahl an Dimensionen des Embeddings
        :param reuse: bool, falls das Netzwerk zum Testen auch benutzt werden soll
        :return: Output des letzten Layers für die Logits im TFModel
        """

        # für den Dropout
        is_training = True
        if reuse:
            is_training = False

        data_embed = tf.Variable(tf.zeros([batch_size, max_seq_len, num_dimensions]), dtype=tf.float32)
        data_embed = tf.nn.embedding_lookup(word_vec_embed, input_tensor)

        with tf.variable_scope('ConvNet', reuse=reuse):
            conv1 = tf.layers.conv1d(
                inputs=data_embed,
                filters=64,
                kernel_size=3,
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling1d(
                inputs=conv1,
                pool_size=2,
                strides=2)

            conv2 = tf.layers.conv1d(
                inputs=pool1,
                filters=128,
                kernel_size=3,
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling1d(
                inputs=conv2,
                pool_size=2,
                strides=2)

            conv3 = tf.layers.conv1d(
                inputs=pool2,
                filters=256,
                kernel_size=3,
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling1d(
                inputs=conv3,
                pool_size=2,
                strides=2)

            conv4 = tf.layers.conv1d(
                inputs=pool3,
                filters=512,
                kernel_size=3,
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool4 = tf.layers.max_pooling1d(
                inputs=conv4,
                pool_size=2,
                strides=2)

            conv5 = tf.layers.conv1d(
                inputs=pool4,
                filters=1024,
                kernel_size=3,
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool5 = tf.layers.max_pooling1d(
                inputs=conv5,
                pool_size=2,
                strides=2)

            conv6 = tf.layers.conv1d(
                inputs=pool5,
                filters=2048,
                kernel_size=3,
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool6 = tf.layers.max_pooling1d(
                inputs=conv6,
                pool_size=2,
                strides=2)

            pool_flat = tf.reshape(pool6, [-1, 3*2048])

            pool_flat = tf.cond(just_train_last_layers, lambda: tf.stop_gradient(pool_flat, name='stop_grad'),
                                lambda: pool_flat)

            dropout = tf.layers.dropout(
                inputs=pool_flat,
                rate=0.5,
                training=is_training
            )
            dense = tf.layers.dense(dropout, 3, activation=None)
            print(dense)
        return dense

    def get_LSTM(self, input_tensor, word_vec_embed, just_train_last_layers,
                batch_size, max_seq_len, num_dimensions, reuse, lstm_units=64):
        """
        Inspiriert durch:
        https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
        Struktur mit dynamic_rnn, die die LSTM units + einen Dropout beinhaltet
        :param lstm_units: int Anzahl der LSTM Units, die Verwendet werden
        :param input_tensor: Input des Netzwerks (X_train)
        :param word_vec_embed:  Embedding Matrix
        :param just_train_last_layers: placeholder, bool falls nur der letzte Layer trainiert werden soll
        :param batch_size: int Batchsize
        :param max_seq_len: int Maximale länge der einzelnen Vektoren vom Input
        :param num_dimensions: int Anzahl an Dimensionen des Embeddings
        :param reuse: bool, falls das Netzwerk zum Testen auch benutzt werden soll
        :return: Output des letzten Layers für die Logits im TFModel
        """

        data_embed = tf.Variable(tf.zeros([batch_size, max_seq_len, num_dimensions]), dtype=tf.float32)
        data_embed = tf.nn.embedding_lookup(word_vec_embed, input_tensor)

        with tf.variable_scope('lstm_net', reuse=reuse):
            lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

            rnn_lstm, _ = tf.nn.dynamic_rnn(lstmCell, data_embed, dtype=tf.float32)

            flat_lstm = tf.reshape(rnn_lstm, [-1, 200 * 64])

            cond_flat = tf.cond(just_train_last_layers, lambda: tf.stop_gradient(flat_lstm, name='stop_grad'),
                                lambda: flat_lstm)

            dense = tf.layers.dense(cond_flat, 3, activation=None)
            print(dense)
        return dense

    def get_only_embed(self, input_tensor, word_vec_embed,
                       batch_size, max_seq_len, num_dimensions, reuse):
        """
        Netzwerk Struktur mit Embedding Lookup und Dense Output
        :param input_tensor: Input des Netzwerks (X_train)
        :param word_vec_embed:  Embedding Matrix
        :param batch_size: int Batchsize
        :param max_seq_len: int Maximale länge der einzelnen Vektoren vom Input
        :param num_dimensions: int Anzahl an Dimensionen des Embeddings
        :param reuse: bool, falls das Netzwerk zum Testen auch benutzt werden soll
        :return: Output des letzten Layers für die Logits im TFModel
        """
        data_embed = tf.Variable(tf.zeros([batch_size, max_seq_len, num_dimensions]), dtype=tf.float32)
        data_embed = tf.nn.embedding_lookup(word_vec_embed, input_tensor)
        with tf.variable_scope('Dense', reuse=reuse):
            flat = tf.reshape(data_embed, [-1, 200 * 50])
            dense = tf.layers.dense(flat, 3, activation=None)
            return dense

    def get_without_embed(self, input_tensor, just_train_last_layers, reuse):
        """
        gibt die CNN Struktur zurück
        mit Dense [Batchsize, 200, 50] + 6 mal (Conv + Pooling) + Dropout + Output Dense
        :param input_tensor: Input des Netzwerks (X_train)
        :param just_train_last_layers: placeholder, bool falls nur der letzte Layer trainiert werden soll
        :param reuse: bool, falls das Netzwerk zum Testen auch benutzt werden soll
        :return: Output des letzten Layers für die Logits im TFModel
        """
        is_training = True
        if reuse:
            is_training = False

        with tf.variable_scope('ConvNet', reuse=reuse):
            dense_input = tf.layers.dense(
                tf.to_float(input_tensor),
                (200 * 50),
                activation=None)

            dense_input_re = tf.reshape(dense_input, [-1, 200, 50])

            conv1 = tf.layers.conv1d(
                inputs=dense_input_re,
                filters=64,
                kernel_size=[3],
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool1 = tf.layers.max_pooling1d(
                inputs=conv1,
                pool_size=[2],
                strides=2)

            conv2 = tf.layers.conv1d(
                inputs=pool1,
                filters=128,
                kernel_size=[3],
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool2 = tf.layers.max_pooling1d(
                inputs=conv2,
                pool_size=[2],
                strides=2)

            conv3 = tf.layers.conv1d(
                inputs=pool2,
                filters=256,
                kernel_size=[3],
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool3 = tf.layers.max_pooling1d(
                inputs=conv3,
                pool_size=[2],
                strides=2)

            conv4 = tf.layers.conv1d(
                inputs=pool3,
                filters=512,
                kernel_size=[3],
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool4 = tf.layers.max_pooling1d(
                inputs=conv4,
                pool_size=[2],
                strides=2)

            conv5 = tf.layers.conv1d(
                inputs=pool4,
                filters=1024,
                kernel_size=[3],
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool5 = tf.layers.max_pooling1d(
                inputs=conv5,
                pool_size=[2],
                strides=2)

            conv6 = tf.layers.conv1d(
                inputs=pool5,
                filters=2048,
                kernel_size=[3],
                padding="same",
                strides=1,
                activation=tf.nn.relu)

            pool6 = tf.layers.max_pooling1d(
                inputs=conv6,
                pool_size=[2],
                strides=2)

            pool_flat = tf.reshape(pool6, [-1, 3 * 2048])

            pool_flat = tf.cond(just_train_last_layers, lambda: tf.stop_gradient(pool_flat, name='stop_grad'),
                                lambda: pool_flat)

            dropout = tf.layers.dropout(
                inputs=pool_flat,
                rate=0.5,
                training=is_training
            )
            dense = tf.layers.dense(dropout, 3, activation=None)
            print(dense)
            return dense
