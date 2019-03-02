import tensorflow as tf
import numpy as np
import datetime
from sklearn.metrics import precision_recall_fscore_support as score
from classes.structure import Structure


class TFModel:
    
    def __init__(self, input_dataset, word_vec_embed, net_name, epochs=4,
                 batch_size=100, max_seq_len=200, num_dimensions=50, restore=False):
        """
        init TFModel
        initalisiert das Netz und das Dataset
        :param input_dataset: Input Dataset für das Netz
        :param word_vec_embed: Embeddinf für den lookup
        :param net_name: String Name des Netzes, Netz wird danach geladen
        :param epochs: int Anzahl der Trainings durchgänge
        :param batch_size: int Batchsize
        :param max_seq_len: int Maximale Länge des Embeddings, fixe größe des Netzes
        :param num_dimensions: int Dimenson des Embeddings
        :param restore: bool falls das Netzwerk geladen werden soll
        """

        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        tf.reset_default_graph()
        structure = Structure(net_name)
        self.net_name = net_name
        self.epochs = epochs
        self.batch_size = batch_size
        (X_train, y_train), (X_val, y_val) = input_dataset

        self.graph = tf.Graph()
        if not restore:
            with self.graph.as_default():
                self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

                self.just_train_last_layers = tf.placeholder_with_default(False, [], name='just_train_last_layers')
                self.ph_data = tf.placeholder(tf.int32, [None, max_seq_len], name='ph_data')
                self.ph_labels = tf.placeholder(tf.int32, [None, 3], name='ph_labels')

                train_iterator, self.train_data_init_op = self.load_data_into_dataset('train')
                val_iterator, self.val_data_init_op = self.load_data_into_dataset('val')

                input_data, labels = train_iterator.get_next()
                input_val_data, val_labels = val_iterator.get_next()

                input_data = tf.cast(input_data, tf.int32)
                (self.X_train, self.y_train) = (X_train, y_train)
                (self.X_val, self.y_val) = (X_val, y_val)

                logits = structure.get_main_net(input_data, word_vec_embed, self.just_train_last_layers,
                                                batch_size=batch_size,
                                                max_seq_len=max_seq_len,
                                                num_dimensions=num_dimensions,
                                                reuse=False)

                logits_val = structure.get_main_net(input_val_data, word_vec_embed, self.just_train_last_layers,
                                                    batch_size=batch_size,
                                                    max_seq_len=max_seq_len,
                                                    num_dimensions=num_dimensions,
                                                    reuse=True)
                # values for F1 Score
                self.val_labels_arg = tf.argmax(val_labels, 1, name="arg_labels_val")
                self.logits_val_arg = tf.argmax(logits_val, 1, name="arg_logits_val")
                self.labels_arg = tf.argmax(labels, 1, name="arg_labels")
                self.logits_arg = tf.argmax(logits, 1, name="arg_logits")

                correct_pred = tf.equal(self.logits_arg, self.labels_arg)

                val_correct_pred = tf.equal(self.logits_val_arg, self.val_labels_arg)

                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                self.val_accuracy = tf.reduce_mean(tf.cast(val_correct_pred, tf.float32))

                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

                with tf.variable_scope('optimizer'):
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        else:
            with self.graph.as_default():
                with tf.Session(graph=self.graph, config=config) as sess:
                    self.restor_old_model(sess)

                    self.learning_rate = self.graph.get_tensor_by_name('learning_rate:0')

                    self.ph_data = self.graph.get_tensor_by_name('ph_data:0')
                    self.ph_labels = self.graph.get_tensor_by_name('ph_labels:0')
                    self.just_train_last_layers = self.graph.get_tensor_by_name('just_train_last_layers:0')

                    (X_train, y_train), (X_val, y_val) = input_dataset

                    self.train_data_init_op = self.graph.get_operation_by_name('dataset_init_train')

                    self.val_data_init_op = self.graph.get_operation_by_name('dataset_init_val')

                    (self.X_train, self.y_train) = (X_train, y_train)
                    (self.X_val, self.y_val) = (X_val, y_val)

                    self.accuracy = self.graph.get_tensor_by_name('Mean:0')

                    self.val_accuracy = self.graph.get_tensor_by_name('Mean_1:0')

                    self.val_labels_arg = self.graph.get_tensor_by_name('arg_labels_val:0')
                    self.logits_val_arg = self.graph.get_tensor_by_name('arg_logits_val:0')
                    self.labels_arg = self.graph.get_tensor_by_name('arg_labels:0')
                    self.logits_arg = self.graph.get_tensor_by_name('arg_logits:0')

                    self.loss = self.graph.get_tensor_by_name('Mean_2:0')
                    self.optimizer = self.graph.get_operation_by_name('optimizer/Adam')
        
    def load_data_into_dataset(self, name):
        """
        initialisier und lädt Dataset

        muss ausgeführt werden um Daten in das Netzwerk zu bekommen:
        #####
        sess.run(dataset_init_op, feed_dict={self.ph_data: X,
                                                     self.ph_labels: y})
        #####
        :param name: String val oder train, damit die verschiedenen Datasets auch geladen werden können
        :return: iterator Netz input, dataset_init_op zum initialisieren
        """

        dataset = tf.data.Dataset.from_tensor_slices((self.ph_data, self.ph_labels))
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.shuffle(buffer_size=100000)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        dataset_init_op = iterator.make_initializer(dataset, name=str('dataset_init_' + name))

        return iterator, dataset_init_op

    def restor_old_model(self, sess):
        """
        lädt altes Model

        :param sess: TF Session
        """
        print('\nRestoring...')
        saver = tf.train.import_meta_graph("models/"+self.net_name+"/pretrained_model.ckpt-0.meta")
        print(tf.train.latest_checkpoint('models/' + self.net_name))
        saver.restore(sess, tf.train.latest_checkpoint('models/'+self.net_name))

    def save_model(self, sess, i):
        """
        speichert Model in models/net_name/ ab

        :param sess: TF Session
        :param i: Iterationsschritt zum abspeichern
        """
        # Save model state
        print('\nSaving...')
        saver = tf.train.Saver()
        save_path = saver.save(sess, "models/"+self.net_name+"/pretrained_model.ckpt", global_step=i)
        print("saved to %s" % save_path)

    def get_mean_of_score_list(self, score_list):
        """
        rechnet den durchschnittlichen Scores für jedes Label aus

        :param score_list: liste mit den Score Array
        :return: array mit mean Score
        """
        score = np.array([0., 0., 0.])
        count = 0
        for i in score_list:
            score[0] += i[0]
            score[1] += i[1]
            score[2] += i[2]
            count += 1
        print(count, "\t", score)
        score = score / count

        return score

    def train_model(self, scale=1, learn_rate=0.001, save=False):
        """
        Trainiert das initialisierte Netz
        :param scale: Skalierung wie viele Schritte geprinted bzw. in Tensorboard geschrieben werden sollen
        :param learn_rate: Lernrate default 0.001 für Adam
        :param save: bool ob gespeichert werden soll
        :return:
        """
        return_dict = dict()
        test_acc_list = []
        support_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        loss_list = []
        acc_list = []

        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.graph, config=config) as sess:
            tf.global_variables_initializer().run(session=sess)
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Accuracy', self.accuracy)

            merged = tf.summary.merge_all()
            logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            writer = tf.summary.FileWriter(logdir, sess.graph)

            i = 0

            for epoch in range(self.epochs):
                tf.local_variables_initializer().run(session=sess)
                # trainings Daten werden in das Dataset geladen
                sess.run(self.train_data_init_op, feed_dict={self.ph_data: self.X_train,
                                                             self.ph_labels: self.y_train})

                # save 0 Iteration
                if i == 0 and save:
                    self.save_model(sess, i)
                # E
                while True:
                    try:
                        sess.run(self.optimizer, feed_dict={self.learning_rate: learn_rate})
                        # Write summary to Tensorboard
                        if i % (scale*20) == 0 or i==0:
                            loss, acc, summary = sess.run([self.loss, self.accuracy, merged])
                            print(i, ": loss: ", loss, "\t acc: ", acc)
                            writer.add_summary(summary, i)
                            loss_list.append(loss)
                            acc_list.append(acc)

                    except tf.errors.OutOfRangeError:
                        break
                    i += 1
                if save:
                    self.save_model(sess, i)
                sess.run(self.val_data_init_op, feed_dict={self.ph_data: self.X_val,
                                                           self.ph_labels: self.y_val})

                test_acc_list = []
                support_list = []
                f1_list = []
                precision_list = []
                recall_list = []
                while True:
                    try:
                        val_logits, test_acc, val_labels = sess.run([self.logits_val_arg, self.val_accuracy,
                                                                     self.val_labels_arg])
                        test_acc_list.append(test_acc)

                        precision, recall, f1, support = score(val_labels, val_logits)

                        if f1.shape[0] is 3:
                            f1_list.append(f1)
                            precision_list.append(precision)
                            recall_list.append(recall)
                            test_acc_list.append(test_acc)
                            support_list.append(support)
                        else:
                            print(f1.shape)
                    except tf.errors.OutOfRangeError:
                        break
                print()
                print(epoch, "\tval accuracy: ", np.mean(test_acc_list), "\t f_1 score: ",
                      self.get_mean_of_score_list(f1_list))
                print()

        return_dict['val_acc'] = test_acc_list
        return_dict['f1_list'] = self.get_mean_of_score_list(f1_list)
        return_dict['precision_list'] = self.get_mean_of_score_list(precision_list)
        return_dict['recall_list'] = self.get_mean_of_score_list(recall_list)
        return_dict['support'] = self.get_mean_of_score_list(support_list)
        return_dict['acc'] = acc_list
        return_dict['loss'] = loss_list
        return return_dict

    def train_last_layer(self, scale=1, learn_rate=0.001, epochs=1):
        """
        trainiert den letzten Layer eines abgespeicherten Netzwerks und testet danach

        :param scale: int/float Skalierungs Faktor zum printen des zwischen Acc und Loss ggf. Tensorboard
        :param learn_rate: float Lernrate beim trainieren
        :param epochs: int Anzahl der Epochen, da nur der letzte Layer trainiert wird, wird ein Standard von 1 verwendet
        :return: dict (val_acc, f1_score, precision_list, recall_list, support)
        """
        return_dict = dict()
        test_acc_list = []
        support_list = []
        f1_list = []
        precision_list = []
        recall_list = []

        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.graph, config=config) as sess:
            tf.global_variables_initializer().run(session=sess)

            i = 0
            for epoch in range(epochs):
                tf.local_variables_initializer().run(session=sess)
                sess.run(self.train_data_init_op, feed_dict={self.ph_data: self.X_train,
                                                             self.ph_labels: self.y_train})

                while True:
                    try:
                        sess.run(self.optimizer, feed_dict={self.learning_rate: learn_rate,
                                                            self.just_train_last_layers: True})

                        if i % (scale * 20) == 0 or i == 0:
                            loss, acc, = sess.run([self.loss, self.accuracy])
                            print(i, ": loss: ", loss, "\t acc: ", acc)

                    except tf.errors.OutOfRangeError:
                        break
                    i += 1

                sess.run(self.val_data_init_op, feed_dict={self.ph_data: self.X_val,
                                                           self.ph_labels: self.y_val})

                test_acc_list = []
                support_list = []
                f1_list = []
                precision_list = []
                recall_list = []
                while True:
                    try:
                        val_logits, test_acc, val_labels = sess.run(
                            [self.logits_val_arg, self.val_accuracy, self.val_labels_arg])
                        print("acc: ", test_acc)
                        test_acc_list.append(test_acc)

                        precision, recall, f1, support = score(val_labels, val_logits)

                        if f1.shape[0] is 3:
                            f1_list.append(f1)
                            precision_list.append(precision)
                            recall_list.append(recall)
                            test_acc_list.append(test_acc)
                            support_list.append(support)
                        else:
                            print(f1.shape)
                    except tf.errors.OutOfRangeError:
                        break
                print()
                print(epoch, "\tval accuracy: ", np.mean(test_acc_list), "\t f_1 score: ",
                      self.get_mean_of_score_list(f1_list))
                print()

        return_dict['val_acc'] = test_acc_list
        return_dict['f1_list'] = self.get_mean_of_score_list(f1_list)
        return_dict['precision_list'] = self.get_mean_of_score_list(precision_list)
        return_dict['recall_list'] = self.get_mean_of_score_list(recall_list)
        return_dict['support'] = self.get_mean_of_score_list(support_list)

        return return_dict

    def just_test(self):
        """
        testet ein abgespeichertes Netzwerk

        :return: dict (val_acc, f1_score, precision_list, recall_list, support)
        """
        return_dict = dict()
        test_acc_list = []
        support_list = []
        f1_list = []
        precision_list = []
        recall_list = []

        config = tf.ConfigProto()

        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.graph, config=config) as sess:
            tf.global_variables_initializer().run(session=sess)
            sess.run(self.val_data_init_op, feed_dict={self.ph_data: self.X_val,
                                                       self.ph_labels: self.y_val})

            while True:
                try:
                    val_logits, test_acc, val_labels = sess.run(
                        [self.logits_val_arg, self.val_accuracy, self.val_labels_arg])
                    print("acc: ", test_acc)
                    test_acc_list.append(test_acc)

                    precision, recall, f1, support = score(val_labels, val_logits)

                    if f1.shape[0] is 3:
                        f1_list.append(f1)
                        precision_list.append(precision)
                        recall_list.append(recall)
                        test_acc_list.append(test_acc)
                        support_list.append(support)
                    else:
                        print(f1.shape)
                except tf.errors.OutOfRangeError:
                    break
            print()
            print("val accuracy: ", np.mean(test_acc_list), "\t f_1 score: ",
                  self.get_mean_of_score_list(f1_list))
            print()

        return_dict['val_acc'] = test_acc_list
        return_dict['f1_list'] = self.get_mean_of_score_list(f1_list)
        return_dict['precision_list'] = self.get_mean_of_score_list(precision_list)
        return_dict['recall_list'] = self.get_mean_of_score_list(recall_list)
        return_dict['support'] = self.get_mean_of_score_list(support_list)

        return return_dict
