import os.path

import numpy as np
from keras import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.metrics import Precision, Recall
from keras_nlp.metrics import EditDistance
#from keras_nlp.metrics import Perplexity

from data_processor import DataProcessor
from quality_evaluator import QualityEvaluatorCallback
from song_generator import GeneratorCallback


class DefaultLSTMModel:
    def __init__(self, X, y, v_X, v_y):
        self.keras_model = Sequential()
        self.keras_model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, dropout=0.2))
        self.keras_model.add(LSTM(256, dropout=0.2))
        self.keras_model.add(Dense(y.shape[1], activation='softmax'))
        self.keras_model.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy', Precision(), Recall(name="recall")],
                                 optimizer='adam')
        self.X = X
        self.y = y
        self.validation_X = v_X
        self.validation_y = v_y

    def train(self, generator=None, max_epochs=512):
        filepath = os.path.join("trained_models", "default-lstm-weights.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor="loss", min_delta=0.01, patience=64)
        callbacks_list = [checkpoint, early_stop, CSVLogger('default_lstm.log')]

        if generator:
            callbacks_list.append(GeneratorCallback(generator))

        self.keras_model.fit(self.X, self.y, epochs=max_epochs, batch_size=64, callbacks=callbacks_list,
                             validation_data=(self.validation_X, self.validation_y))


class BetaLSTMModel:
    def __init__(self, X, y, v_X, v_y):
        self.keras_model = Sequential()
        self.keras_model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, dropout=0.36))
        self.keras_model.add(LSTM(256, dropout=0.2))
        self.keras_model.add(Dense(y.shape[1], activation='softmax'))
        self.keras_model.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy', Precision(), Recall(name="recall")],
                                 optimizer='adam')
        self.X = X
        self.y = y
        self.validation_X = v_X
        self.validation_y = v_y

    def train(self, generator=None, max_epochs=512):
        filepath = os.path.join("trained_models", "beta-lstm-weights.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor="loss", min_delta=0.01, patience=64)
        callbacks_list = [checkpoint, early_stop, CSVLogger('beta_lstm.log')]

        if generator:
            callbacks_list.append(GeneratorCallback(generator))

        self.keras_model.fit(self.X, self.y, epochs=max_epochs, batch_size=64, callbacks=callbacks_list,
                             validation_data=(self.validation_X, self.validation_y))


class GammaLSTMModel:
    def __init__(self, X, y, v_X, v_y):
        self.keras_model = Sequential()
        self.keras_model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, dropout=0.2))
        self.keras_model.add(LSTM(512, dropout=0.2))
        self.keras_model.add(Dense(y.shape[1], activation='softmax'))
        self.keras_model.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy', Precision(), Recall(name="recall")],
                                 optimizer='adam')
        self.X = X
        self.y = y
        self.validation_X = v_X
        self.validation_y = v_y

    def train(self, generator=None, max_epochs=512):
        filepath = os.path.join("trained_models", "gamma-lstm-weights.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor="loss", min_delta=0.01, patience=64)
        #quality_eval = QualityEvaluatorCallback(self.validation_X, self.validation_y, self.keras_model)
        callbacks_list = [checkpoint, early_stop, CSVLogger('gamma_lstm.log')]

        if generator:
            callbacks_list.append(GeneratorCallback(generator))

        self.keras_model.fit(self.X, self.y, epochs=max_epochs, batch_size=64, callbacks=callbacks_list,
                             validation_data=(self.validation_X, self.validation_y))


class DefaultGRUModel:
    def __init__(self, X, y, v_X, v_y):
        self.keras_model = Sequential()
        self.keras_model.add(GRU(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, dropout=0.2))
        self.keras_model.add(GRU(256, dropout=0.2))
        self.keras_model.add(Dense(y.shape[1], activation='softmax'))
        self.keras_model.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy', Precision(), Recall(name="recall")],
                                 optimizer='adam')
        self.X = X
        self.y = y
        self.validation_X = v_X
        self.validation_y = v_y

    def train(self, generator=None, max_epochs=512):
        filepath = os.path.join("trained_models", "default-gru-weights.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor="loss", min_delta=0.01, patience=64)
        callbacks_list = [checkpoint, early_stop, CSVLogger('default_gru.log')]

        if generator:
            callbacks_list.append(GeneratorCallback(generator))

        self.keras_model.fit(self.X, self.y, epochs=max_epochs, batch_size=64, callbacks=callbacks_list,
                             validation_data=(self.validation_X, self.validation_y))


class BetaGRUModel:
    def __init__(self, X, y, v_X, v_y):
        self.keras_model = Sequential()
        self.keras_model.add(GRU(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, dropout=0.2))
        self.keras_model.add(GRU(128, dropout=0.2))
        self.keras_model.add(Dense(y.shape[1], activation='softmax'))
        self.keras_model.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy', Precision(), Recall(name="recall")],
                                 optimizer='adam')
        self.X = X
        self.y = y
        self.validation_X = v_X
        self.validation_y = v_y

    def train(self, generator=None, max_epochs=512):
        filepath = os.path.join("trained_models", "beta-gru-weights.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor="loss", min_delta=0.01, patience=64)
        callbacks_list = [checkpoint, early_stop, CSVLogger('beta_gru.log')]

        if generator:
            callbacks_list.append(GeneratorCallback(generator))

        self.keras_model.fit(self.X, self.y, epochs=max_epochs, batch_size=64, callbacks=callbacks_list,
                             validation_data=(self.validation_X, self.validation_y))


class OmicronLSTM:
    def __init__(self, X, y, data_processor: DataProcessor, v_X, v_y):
        self.keras_model = Sequential()

        embeddings_index = {}
        f = open("glove.6B.100d.txt")
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefficients
        f.close()

        word_index = data_processor.tokenizer.word_index
        max_length = data_processor.max_length(data_processor.train_lyrics)

        embedding_matrix = np.random.normal(0, 1, size=(len(word_index) + 1, 100))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(input_dim=len(word_index) + 1,
                                    output_dim=100,
                                    weights=[embedding_matrix],
                                    input_length=max_length,
                                    trainable=True)

        self.keras_model.add(embedding_layer)
        self.keras_model.add(LSTM(256, input_shape=(100, 1), return_sequences=True, dropout=0.2))
        self.keras_model.add(LSTM(256, dropout=0.2))
        self.keras_model.add(Dense(y.shape[1], activation='softmax'))
        self.keras_model.compile(loss='categorical_crossentropy',
                                 metrics=['accuracy', Precision(), Recall(name="recall")],
                                 optimizer='adam')
        self.X = X
        self.y = y
        self.validation_X = v_X
        self.validation_y = v_y

    def train(self, generator=None, max_epochs=512):
        filepath = os.path.join("trained_models", "trained_models/omicron-lstm-weights.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        early_stop = EarlyStopping(monitor="loss", min_delta=0.01, patience=64)
        callbacks_list = [checkpoint, early_stop, CSVLogger('omicron_lstm.log')]

        if generator:
            callbacks_list.append(GeneratorCallback(generator))

        self.keras_model.fit(self.X, self.y, epochs=max_epochs, batch_size=64, callbacks=callbacks_list,
                             validation_data=(self.validation_X, self.validation_y))
