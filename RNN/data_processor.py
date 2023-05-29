import pandas as pd
from keras.preprocessing.text import Tokenizer, one_hot
import numpy as np
from keras.utils import to_categorical


def pad_array(array, length):
    return array + [0] * (length - len(array))


class DataProcessor:
    def __init__(self, csv_filename, seqs_length=None, mode="words"):
        self.lyrics = list(pd.read_csv(csv_filename)["Lyrics"].dropna())
        if mode == "words":
            self.tokenizer = Tokenizer()
        elif mode == "chars":
            self.tokenizer = Tokenizer(char_level=True)
        else:
            raise ValueError("Unsupported mode: " + mode)
        self.mode = mode
        self.tokenizer.fit_on_texts(self.lyrics)
        self.seqs_length = seqs_length
        self.train_lyrics = self.lyrics[:int(0.8 * len(self.lyrics))]
        self.validation_lyrics = self.lyrics[int(0.8 * len(self.lyrics)):]

    def texts_to_ints(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def ints_to_text(self, ints):
        return self.tokenizer.sequences_to_texts([ints])[0]

    def texts_to_onehots(self, texts):
        n = len(self.tokenizer.word_index) + 1
        one_hots = [one_hot(lyric, n) for lyric in texts]
        return one_hots

    def vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def max_length(self, texts):
        full_sequences = self.texts_to_ints(texts)
        return max([len(seq) for seq in full_sequences])

    def training_data(self, kind="ints", padded=False):
        if kind == "ints":
            full_sequences = self.texts_to_ints(self.train_lyrics)
        elif kind == "onehots":
            full_sequences = self.texts_to_onehots(self.train_lyrics)
        else:
            raise ValueError("Kind must be either ints or onehots")

        if not padded and self.seqs_length:
            X = []
            y = []
            for full_sequence in full_sequences:
                for i in range(len(full_sequence) - self.seqs_length):
                    X.append(full_sequence[i:i + self.seqs_length])
                    y.append(full_sequence[i + self.seqs_length])
            X = np.reshape(X, (len(X), len(X[0]), 1))
            # TODO: Do we need that?
            # X = X / float(self.vocab_size())
        elif padded:
            max_length = max([len(seq) for seq in full_sequences])
            X = []
            y = []
            for full_sequence in full_sequences:
                for i in range(len(full_sequence)):
                    X.append(pad_array(full_sequence[:i], max_length))
                    y.append(full_sequence[i])
            X = np.reshape(X, (len(X), len(X[0]), 1))
        else:
            raise ValueError("Can't use padding along with seqs_length")

        y = to_categorical(y, num_classes=self.vocab_size())

        return X, y

    def validation_data(self, kind="ints", padded=False):
        if kind == "ints":
            full_sequences = self.texts_to_ints(self.validation_lyrics)
        elif kind == "onehots":
            full_sequences = self.texts_to_onehots(self.validation_lyrics)
        else:
            raise ValueError("Kind must be either ints or onehots")

        if not padded and self.seqs_length:
            X = []
            y = []
            for full_sequence in full_sequences:
                for i in range(len(full_sequence) - self.seqs_length):
                    X.append(full_sequence[i:i + self.seqs_length])
                    y.append(full_sequence[i + self.seqs_length])
            X = np.reshape(X, (len(X), len(X[0]), 1))
            # TODO: Do we need that?
            # X = X / float(self.vocab_size())
        elif padded:
            max_length = max([len(seq) for seq in full_sequences])
            X = []
            y = []
            for full_sequence in full_sequences:
                for i in range(len(full_sequence)):
                    X.append(pad_array(full_sequence[:i], max_length))
                    y.append(full_sequence[i])
            X = np.reshape(X, (len(X), len(X[0]), 1))
        else:
            raise ValueError("Can't use padding along with seqs_length")

        y = to_categorical(y, num_classes=self.vocab_size())

        return X, y
