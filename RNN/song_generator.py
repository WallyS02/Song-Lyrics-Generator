from random import randint
import random

import keras.callbacks
import numpy as np
from data_processor import DataProcessor


class SongGenerator:
    def __init__(self, data_processor: DataProcessor, model):
        self.data_processor = data_processor
        self.model = model

    def sample(self, predictions, temperature=0.1):
        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        probs = np.random.multinomial(1, predictions, 1)
        return np.argmax(probs)

    def generate(self, tokens_per_line=6, lines=4, temp=1.0, custom_seed=None):
        result_indexes = []

        if not custom_seed:
            seed_idx = randint(0, len(self.model.X) - 1)
            seed = self.model.X[seed_idx]
            seed = list(np.reshape(seed, (len(seed))))
        else:
            seed = custom_seed

        for _ in range(lines * tokens_per_line):
            data_in = np.reshape(seed, (1, len(seed), 1))
            #data_in = data_in / float(self.data_processor.vocab_size())
            prediction = self.model.keras_model.predict(data_in, verbose=0)
            #out_index = np.argmax(prediction)
            
            # r = random.random()
            # curr = 0.0
            # out_index = -1
            #
            # for idx, pred in sorted(enumerate(list(prediction.flatten())), reverse=True, key=lambda x: x[1]):
            #     out_index = idx
            #     curr += pred
            #     if curr >= r:
            #         break

            out_index = self.sample(prediction[0] + 10e-5, temp)
            result_indexes.append(out_index)
            seed.append(out_index)
            seed = seed[1:len(seed)]

        result_tokens = self.data_processor.ints_to_text(result_indexes).split(" ")

        if self.data_processor.mode == "chars":
            raise NotImplementedError()

        # Capitalize I words
        for i, token in enumerate(result_tokens):
            if token == "i":
                result_tokens[i] = token.capitalize()
            elif token == "i'm":
                result_tokens[i] = "I'm"

        result = ""
        for line_idx in range(lines):
            result_tokens[line_idx*tokens_per_line] = result_tokens[line_idx*tokens_per_line].capitalize()
            result += " ".join(result_tokens[line_idx*tokens_per_line:line_idx*tokens_per_line + tokens_per_line]) \
                      + "\n"

        return result.rstrip("\n")


class GeneratorCallback(keras.callbacks.Callback):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        print(self.generator.generate())
