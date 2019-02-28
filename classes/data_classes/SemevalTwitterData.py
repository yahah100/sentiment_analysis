import pandas as pd
import numpy as np
from classes.Cleaner import Cleaner


class SemevalTwitterData:

    def __init__(self, word_index, max_seq_length, create=False):
        """
        init Dataset
        benutzt Cleaner zum Reinigen und Tokenisen der Daten
        :param word_index: word_index Vektor, damit den Wörtern der richtige Index zugewiesen werden kann
        :param max_seq_length: int max länge Wortbegrenzung
        :param create: bool falls, die Zusammengeführt csv Datei noch nicht vorhanden ist.
        """
        if create:
            self.make_dataset_and_save_in_csv()

        self.complete_dataset = self.read_complete_dataset()
        self.complete_dataset = self.complete_dataset.sample(frac=1).reset_index(drop=True)
        self.cleaner = Cleaner()
        X_data = self.make_id_data(word_index, self.complete_dataset['text'], max_seq_length)

        y_data = self.convert_to_labels(self.complete_dataset['sentiment'])

        self.data = self.create_data_set(X_data, y_data)

    def make_dataset_and_save_in_csv(self):
        """
        erstellt csv Datei aus dem Semeval Dataset und dem giant Twitter Dataset
        """
        semeval_data = self.read_semeval_data()
        filename_train = 'train.csv'
        giant_data = self.read_giant_data(filename_train)

        positiv_needed = 21910 - 18909
        negativ_needed = 21910 - 7515

        len_giant_data = len(giant_data['text'])

        x_data_positiv_needed = giant_data['text'][:positiv_needed]

        x_data_negativ_needed = giant_data['text'][(len_giant_data - negativ_needed):]

        for i in range(0, len(x_data_positiv_needed)):
            semeval_data = semeval_data.append({"text": x_data_positiv_needed[i],
                                                "sentiment": "positive"},
                                               ignore_index=True)

        for i in range((len_giant_data - negativ_needed), len_giant_data):
            semeval_data = semeval_data.append({"text": x_data_negativ_needed[i],
                                                "sentiment": "negative"},
                                               ignore_index=True)
        print(semeval_data.shape)
        semeval_data.to_csv("data/twitter/gleichverteilteTwitter.csv", sep='\t', encoding='utf-8')

    def read_semeval_data(self):
        """
        liest Semeval Dateien ein und erstellt pandas Series complete_dataset
        :return: complete_dataset
        """
        tw_names = ['twitter-2013train-A.txt', 'twitter-2013test-A.txt', 'twitter-2013dev-A.txt',
                    'twitter-2014test-A.txt',
                    'twitter-2015train-A.txt', 'twitter-2015test-A.txt',
                    'twitter-2016train-A.txt', 'twitter-2016test-A.txt', 'twitter-2016dev-A.txt',
                    'twitter-2014sarcasm-A.txt']
        my_tw_names = ['train_2013', 'test_2013', 'dev_2013',
                       'test_2014',
                       'train_2015', 'test_2015',
                       'train_2016', 'test_2016', 'dev_2016',
                       'sarcasm_2014']
        tw_dataset = {}
        for i in range(0, 10):
            tw_dataset[my_tw_names[i]] = pd.read_csv(
                'data/twitter/' + tw_names[i], sep="\t",
                header=None, encoding='cp1252')

        # print(tw_dataset['test_2016'].shape)
        tw_dataset['test_2016'] = tw_dataset['test_2016'].drop(columns=[3])
        # print(tw_dataset['test_2016'].shape)

        complete_dataset = tw_dataset['train_2013']

        for i in range(1, 10):
            complete_dataset = complete_dataset.append(tw_dataset[my_tw_names[i]], ignore_index=True)
        # print(complete_dataset.shape)
        complete_dataset = complete_dataset.drop(columns=[0])
        complete_dataset = complete_dataset.rename(columns={1: 'sentiment', 2: 'text'})
        return complete_dataset

    def read_giant_data(self, filename_train):
        """
        liest giant Twitter Data
        :param filename_train: String filename der giant Twitter Data
        :return: Dataset
        """
        tw_data_train = pd.read_csv(
            'data/twitter/' + filename_train, sep=",",
            usecols=[0, 5], names=['sentiment', 'text'], header=None, encoding='cp1252')

        return tw_data_train

    def read_complete_dataset(self):
        """
        liest das schon zusammengeführte Dataset ein
        :return: Dataset
        """
        tw_data_train = pd.read_csv(
            'data/twitter/gleichverteilteTwitter.csv', sep="\t",
            encoding='utf-8')

        return tw_data_train

    def make_id_data(self, word_list, input_data, max_seq_length=200):
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

        for i in range(0, len(input_data)):
            line = self.cleaner.cleane(str(input_data[i]))
            split = line.split()
            j = 0
            for word in split:
                try:
                    ids[i][j] = word_list[word]
                except KeyError:
                    ids[i][j] = 399999
                    # print(word)
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

        for i in range(0, len(input_data)):
            if input_data[i] == 'negative':
                labels[i] = np.array([1, 0, 0])
            elif input_data[i] == 'neutral':
                labels[i] = np.array([0, 1, 0])
            elif input_data[i] == 'positive':
                labels[i] = np.array([0, 0, 1])
            else:
                print("Achtung: ", input_data[i])
        return labels

    def create_data_set(self, train_data, train_labels, VAL_SIZE=10000):
        """
        Erstellt Dataset

        :param x_data: Train Data
        :param labels: Labels
        :param VAL_SIZE: Size der Validation Data bzw. Test Data
        :return:  (train_data, train_labels), (val_data, val_labels)
        """
        val_data = train_data[:VAL_SIZE]
        train_data = train_data[VAL_SIZE:]

        val_labels = train_labels[:VAL_SIZE]
        train_labels = train_labels[VAL_SIZE:]

        return (train_data, train_labels), (val_data, val_labels)

    """
    Getter für das Dataset und rohe Daten
    """

    def get_data(self):
        return self.data

    def get_raw_data(self):
        return self.complete_dataset['sentiment'], self.complete_dataset['text']