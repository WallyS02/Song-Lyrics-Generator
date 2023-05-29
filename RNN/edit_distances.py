import os
from random import choice, randint

import numpy as np
from keras_nlp.src.metrics import EditDistance

from data_processor import DataProcessor
from models import BetaGRUModel, DefaultLSTMModel, GammaLSTMModel, DefaultGRUModel, OmicronLSTM, BetaLSTMModel
from song_generator import SongGenerator

dp = DataProcessor("ac_dc.csv", 10)
X, y = dp.training_data()
v_X, v_y = dp.validation_data()

# dp = DataProcessor("ac_dc.csv", None)
# X, y = dp.training_data("ints", True)
# v_X, v_y = dp.validation_data("ints", True)
model = GammaLSTMModel(X, y, v_X, v_y)
weights_to_load = "gamma-lstm-weights.hdf5"
weights_path = os.path.join("trained_models", weights_to_load)
model.keras_model.load_weights(weights_path)

lines = 1
words_in_line = 1
generator = SongGenerator(dp, model)
eds = []

for i in range(16):
    original_text = choice(dp.validation_lyrics).split(" ")
    original_begin = randint(0, len(original_text) - lines * words_in_line - 1)
    original_end = original_begin + lines * words_in_line + 10
    original = " ".join(original_text[original_begin:original_end])

    original_ints = dp.texts_to_ints([original])[0]
    original = original_ints[10:10+lines*words_in_line]

    seed = original_ints[:10]

    generated = generator.generate(words_in_line, lines, custom_seed=seed).lower()
    generated = dp.texts_to_ints([generated])[0]

    ed = EditDistance(normalize=True)
    eds.append(ed(generated, original).numpy())

print(f"Average ED: {np.average(eds)}")
print(f"Min ED: {np.min(eds)}")
print(f"Max ED: {np.max(eds)}")
