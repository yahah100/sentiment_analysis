import numpy as np
import pandas as pd


class MedicalData:

    def __init__(self, word_index, max_seq_length):
        """
        init Dataset
        :param word_index: word_index Vektor, damit den Wörtern der richtige Index zugewiesen werden kann
        :param max_seq_length: int max länge Satzbegrenzung
        """
        filename = 'medicalData'
        self.med_raw_data = self.read_data(filename)

        x_data = self.make_id_data(word_index, self.med_raw_data[2], max_seq_length)
        y_data = self.convert_to_labels(self.med_raw_data[1])

        self.data = self.create_data_set(x_data, y_data)

    def read_data(self, filename):

        med_data = pd.read_csv(
                'data/medical/' + filename, sep="\t",
                header=None, encoding='utf8')

        print(med_data.shape)
        return med_data

    def make_id_data(self, word_list, input_data, max_seq_length=500):
        """
        erstellt die X Train Daten
        wandelt Wörter in die  Embedding ids um
        :param word_list: word_index Vektor, damit den Wörtern der richtige Index zugewiesen werden kann
        :param input_data: X_train Tweets
        :param max_seq_length: int max länge Wortbegrenzung
        :return: Matrix mit Embedding ids mit der länge max_seq_length
        """
        print(len(input_data))
        ids = np.zeros((len(input_data), max_seq_length), dtype='int32')

        for i in range(1, len(input_data)):
            line = input_data[i]
            split = line.split()
            j = 0
            for word in split:
                try:
                    ids[i][j] = word_list[word]
                except KeyError:
                    ids[i][j] = 399999  # Vector for
                j += 1
                if j >= max_seq_length:
                    break
        return ids

    def convert_to_labels(self, input_data):
        """
        Mapping auf einheitliche Labels

        :param input_data: Label
        :return: einheitliche Labels
        """
        labels = np.zeros((len(input_data), 3), dtype='int32')

        for i in range(1, len(input_data)):
            input = int(float(input_data[i]))
            if input == 1 or input == 2 or input == 3:
                labels[i] = np.array([1, 0, 0])
            elif input == 4 or input == 5 or input == 6 or input == 7:
                labels[i] = np.array([0, 1, 0])
            elif input == 8 or input == 9 or input == 10:
                labels[i] = np.array([0, 0, 1])
            else:
                print("Achtung: ", input_data[i])
        return labels

    def create_data_set(self, x_data, labels, VAL_SIZE=1400):
        """
        Erstellt Dataset

        :param x_data: Train Data
        :param labels: Labels
        :param VAL_SIZE: Size der Validation Data bzw. Test Data
        :return:  (train_data, train_labels), (val_data, val_labels)
        """
        val_data = x_data[:VAL_SIZE]
        train_data = x_data[VAL_SIZE:]

        val_labels = labels[:VAL_SIZE]
        train_labels = labels[VAL_SIZE:]

        return (train_data, train_labels), (val_data, val_labels)

    """
    Getter für das Dataset und rohe Daten
    """

    def get_raw_data(self):
        return self.med_raw_data

    def get_data(self):
        return self.data