import math

import numpy as np
from keras import Model
from keras.callbacks import Callback
from keras_nlp.src.metrics import EditDistance
from numpy.random import choice


class QualityEvaluatorCallback(Callback):
    def __init__(self, validation_X, validation_y):
        super().__init__()
        self.validation_X = validation_X
        self.validation_y = validation_y

    def on_epoch_end(self, epoch, logs=None):
        print(f"{epoch} Calculating quality metrics!")
        perplexity = logs.get("loss")**math.e
        print(f"Perplexity: {perplexity}")

        # true_positives = {}
        # false_positives = {}
        # false_negatives = {}
        #
        # v_set_size = self.validation_X.shape[0]
        # random_ids = np.random.randint(0, v_set_size, size=100)
        # validation_X_subset = self.validation_X[random_ids]
        # validation_y_subset = self.validation_y[random_ids]
        #
        # i = 0
        # for X, label in zip(validation_X_subset, validation_y_subset):
        #     prediction = self.keras_model.predict(X, verbose=0)
        #
        #     predicted_idx = np.argmax(prediction)
        #     valid_idx = np.argmax(label)
        #
        #     if predicted_idx not in true_positives:
        #         true_positives[predicted_idx] = 0
        #     if predicted_idx not in false_positives:
        #         false_positives[predicted_idx] = 0
        #     if predicted_idx not in false_negatives:
        #         false_negatives[predicted_idx] = 0
        #
        #     if valid_idx not in true_positives:
        #         true_positives[valid_idx] = 0
        #     if valid_idx not in false_positives:
        #         false_positives[valid_idx] = 0
        #     if valid_idx not in false_negatives:
        #         false_negatives[valid_idx] = 0
        #
        #     if predicted_idx == valid_idx:
        #         true_positives[predicted_idx] += 1
        #     else:
        #         false_positives[predicted_idx] += 1
        #         false_negatives[valid_idx] += 1
        #
        # precisions = []
        # for tp, fp in zip(true_positives.values(), false_positives.values()):
        #     precisions.append(tp/(tp + fp + 10e-5))
        #
        # mean_precision = np.average(precisions)
        # print(f"Mean precision: {mean_precision}")
