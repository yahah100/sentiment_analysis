import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from classes.Cleaner import Cleaner

class AmazonReviewData:

    def __init__(self, word_index, max_seq_length, create_csv=False):
        """
        init Dataset
        benutzt Cleaner zum Reinigen und Tokenisen der Daten
        :param word_index: word_index Vektor, damit den Wörtern der richtige Index zugewiesen werden kann
        :param max_seq_length: int max länge Satzbegrenzung
        :param create_csv: bool falls, nur die .json Datei vorhanden ist, wird eine csv erstellt, die später geladen werden kann
        """
        cleaner = Cleaner()
        amazon_data_name = 'gleichverteil.csv'

        if create_csv:
            filename = 'raw_amazon_review_alt'

            self.tok = WordPunctTokenizer()

            amazon_raw_data_rev, amazon_raw_data_score = self.read_data(filename)

            print(amazon_raw_data_score.dtype)
            for i in range(len(amazon_raw_data_rev)):
                amazon_raw_data_rev[i] = cleaner.cleane(amazon_raw_data_rev[i])

            # Verkleine
            amazon_data = amazon_raw_data_score.join(amazon_raw_data_rev, on=0, how='left', lsuffix='_left', rsuffix='_right')

            amazon_data = amazon_data.drop(columns='0_left')
            amazon_data = amazon_data.drop(columns='0_right')
            neg_1 = amazon_data.loc[amazon_data['1_left'] == 1]
            neg_2 = amazon_data.loc[amazon_data['1_left'] == 2]

            neu_3 = amazon_data.loc[amazon_data['1_left'] == 3]

            pos_4 = amazon_data.loc[amazon_data['1_left'] == 4]
            pos_5 = amazon_data.loc[amazon_data['1_left'] == 5]
            pos = pos_4.append(pos_5)
            pos = pos.sample(frac=0.13).reset_index(drop=True)
            neg = neg_1.append(neg_2)
            pos = pos.append(neg)
            pos = pos.append(neu_3)
            pos = pos.sample(frac=1).reset_index(drop=True)

            pos.to_csv("data/amazon_review/gleichverteil.csv", sep='\t',
                       encoding='utf-8')

        amazon_data = pd.read_csv(
            'data/amazon_review/' + amazon_data_name, sep='\t',
            header=None)

        x_data = self.make_id_data(word_index, amazon_data[2], max_seq_length)
        y_data = self.convert_to_labels(amazon_data[1])
        self.raw_data = amazon_data[1], amazon_data[2]
        self.data = self.create_data_set(x_data, y_data)

    def read_data(self, filename):

        amazon_data = pd.read_json(
                'data/amazon_review' + filename,
                lines=True)

        print(amazon_data['reviewText'].shape)
        print(amazon_data['overall'].shape)
        return amazon_data['reviewText'], amazon_data['overall']

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
            line = str(input_data[i])
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
            input = int(input_data[i])
            if input == 1 or input == 2:
                labels[i] = np.array([1, 0, 0])
            elif input == 3:
                labels[i] = np.array([0, 1, 0])
            elif input == 4 or input == 5:
                labels[i] = np.array([0, 0, 1])
            else:
                print("Achtung: ", input_data[i])
        return labels

    def create_data_set(self, x_data, labels, VAL_SIZE=50000):
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
        return self.raw_data

    def get_data(self):
        return self.data
