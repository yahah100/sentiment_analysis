import os
import shutil
import numpy as np
import time

from classes.TFModel import TFModel
from classes.Embedding import Embedding
from classes.data_classes.SemevalTwitterData import SemevalTwitterData
from classes.data_classes.MedicalData import MedicalData
from classes.data_classes.PrimeVideoData import PrimeVideoData
from classes.data_classes.AmazonReviewData import AmazonReviewData

class TestHelper:

    def __init__(self, max_seq_len=200):
        """
        int Helper
        lädt Emebbding

        :param max_seq_len: legt Länge des Embeddings fest
        """
        self.max_seq_len = max_seq_len
        filename = 'data/embedding/glove.twitter.27B.50d.txt'

        embedding = Embedding()

        self.embedding_matrix, embd_length, self.word_to_index = embedding.get_word_embedding(filename)

    def decide_which_input(self, input_name):
        """
        Funtion die, das richtige Dataset lädt nach :param input_name

        :param input_name: String Name der Datenbank die geladen werden soll
        :return: dataset (a_train, label)()
        """
        input_data = None
        if input_name == "Twitter":
            input_data = self.get_tw_data()
        elif input_name == "Medical":
            input_data = self.get_medical_data()
        elif input_name == "Review":
            input_data = self.get_review_data()
        elif input_name == "Prime":
            input_data = self.get_prime_data()
        else:
            print("Wrong input_name!!!")
        return input_data

    def train_input(self, input_name, net_name, epochs=7, print_time=False, restore=False):
        """
        löscht altes gespeichertes Modell
        trainiert neues Model und speichert es unter dem Ordner :param net_name ab

        :param restore:
        :param print_time:
        :param input_name: String Dataset Name
        :param net_name: Netz Name
        :param epochs intr Anzahl der Epochen die trainiert werden soll
        :return: dict loss, accuracy
        """
        print("\n\n ", input_name)
        if not restore:
            self.delete_old_model(net_name)
        input_data = self.decide_which_input(input_name)
        tf_model = TFModel(input_data, self.embedding_matrix, epochs=epochs, net_name=net_name, restore=restore)
        print("---init ready  ", net_name, "---")

        scale = 0.1
        if input_name=="Review":
            scale = 2
        start_time = time.time()
        dict = tf_model.train_model(save=True, scale=scale)
        return_time = time.time() - start_time

        print("--- Test  ", input_name, "---")

        print(np.mean(dict['val_acc']))
        print("f1: ", dict['f1_list'])
        if print_time:
            print("--- %s seconds ---" % (return_time))

        return dict['loss'], dict['acc']

    def hyper_tuning(self, input_name, net_name, learn_rate=0.001, epochs=4):
        """
        Funktion die trainiert wie die train_input Funktion, aber das Netz nicht speichert

        :param input_name:  String Dataset Name
        :param net_name:    String Netz Name
        :param learn_rate:  float Lernrate, damit verschiedene Lernraten ausprobiert werden können
        :param epochs:      int Anzahl der epochen, die trainiert werden soll
        :return:            mean_acc des Test Durchlaufs
        """
        print("\n\n ", input_name)
        input_data = self.decide_which_input(input_name)
        tf_model = TFModel(input_data, self.embedding_matrix, epochs=epochs, net_name=net_name, restore=False)
        print("---init ready  ", net_name, "---")

        # Skalierungs Faktor zum printen des zwischen Acc und Loss ggf. Tensorboard
        scale = 0.1
        if input_name == "Review":
            scale = 2
        dict = tf_model.train_model(save=False, scale=scale, learn_rate=learn_rate)

        print("--- Test  ", input_name, "---")
        print(np.mean(dict['val_acc']))

        return np.mean(dict['val_acc'])

    def just_test(self, input_name, net_name, print_time=False):
        """
        Funktion zum testen eines abgespeicherten Netzwerkes
        restored das abgespeicherte Netzwerk mit dem :param net_name beim
        initalisieren

        :param print_time: bool, ob die Zeit geprinted werden soll, die das trainieren gebraucht hat
        :param input_name: String Dataset Name
        :param net_name:   String Netz Name
        """
        input_data = self.decide_which_input(input_name)
        print("Vorsicht! es muss vorher ein ", net_name, " Netz abgespeichert worden sein")
        tf_model = TFModel(input_data, self.embedding_matrix, net_name=net_name, restore=True)
        start_time = time.time()
        dict = tf_model.just_test()
        return_time = time.time() - start_time
        print("---just Test ", input_name, "---")

        print(np.mean(dict['val_acc']))
        print("f1: ", dict['f1_list'])
        if print_time:
            print("--- %s seconds ---" % return_time)

    def train_last_layer(self, input_name, net_name, epochs=1, print_time=False):
        """
        Funktion zum trainieren des letzten Layer eines abgespeicherten Netzwerkes
        restored das abgespeicherte Netzwerk mit dem :param net_name beim
        initalisieren
        :param print_time: bool, ob die Zeit geprinted werden soll, die das trainieren gebraucht hat
        :param input_name: String Dataset Name
        :param net_name: String Netz Name
        """
        input_data = self.decide_which_input(input_name)
        print("Vorsicht! es muss vorher ein ", net_name, " Netz abgespeichert worden sein")
        tf_model = TFModel(input_data, self.embedding_matrix, net_name=net_name, restore=True)

        scale = 0.1
        if input_name == "Review":
            scale = 5

        start_time = time.time()
        dict = tf_model.train_last_layer(scale=scale, epochs=epochs)
        return_time = time.time() - start_time
        print("---train_last_layer Test ", input_name, "---")

        print(np.mean(dict['val_acc']))
        print("f1: ", dict['f1_list'])
        if print_time:
            print("--- %s seconds ---" % return_time)

    def delete_old_model(self, net_name):
        """
        löscht alten save Ordner der Netzes :param net_name
        :param net_name: String name des zu löschenden Speicher Ordner
        """
        if os.path.exists("models/" + net_name):
            print("delete old models")
            try:
                shutil.rmtree("models/" + net_name)
            except OSError as ex:
                print(ex)
        else:
            print("The file does not exist")


    """
    Getter für die verschiedenen Datasets
    gibt die Daten in der Form (train_data, train_labels), (val_data, val_labels)
    zurück
    """

    def get_tw_data(self):
        tw_data = SemevalTwitterData(self.word_to_index, self.max_seq_len)
        return tw_data.get_data()

    def get_review_data(self):
        amazon_review_data = AmazonReviewData(self.word_to_index, self.max_seq_len)
        return amazon_review_data.get_data()

    def get_medical_data(self):
        med_data = MedicalData(self.word_to_index, self.max_seq_len)
        return med_data.get_data()

    def get_prime_data(self):
        prime_vid_data = PrimeVideoData(self.word_to_index, self.max_seq_len)
        return prime_vid_data.get_data()
