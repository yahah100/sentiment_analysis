import numpy as np

class Embedding:
    """
    Klasse zum Laden des Embeddings
    """

    def get_word_embedding(self, filename, glove_dimension=50):
        """
        lädt das Embedding mit dem Filename
        :param filename: Datei Name der geladenw werden soll
        :param glove_dimension: Dimension des Embeddings
        :return: embedding_matrix, embedding Länge, Wort Index für die Trainingsdaten
        """
        embeddings_index = dict()
        word_index = dict()
        word_to_index = dict()
        count = 0
        f = open(filename)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.array(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            word_index[count] = word
            word_to_index[word] = count
            count += 1
        f.close()

        embd_length = len(embeddings_index)

        embedding_matrix = np.zeros((embd_length, glove_dimension), dtype='float32')
        for i in range(0, embd_length):
            if embeddings_index[word_index[i]].shape[0] == 50:
                embedding_matrix[i] = embeddings_index[word_index[i]]
            else:
                pass
                # print("i:", i, "\t", embeddings_index[word_index[i]].shape)
                # print(word_index[i])

        return embedding_matrix, embd_length, word_to_index