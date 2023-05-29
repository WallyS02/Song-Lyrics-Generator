import os

import models
from data_processor import DataProcessor
from quality_evaluator import QualityEvaluatorCallback
from song_generator import SongGenerator

dp = DataProcessor("ac_dc.csv", 10)
X, y = dp.training_data()
v_X, v_y = dp.validation_data()

print("Models: \ndefault_lstm\nbeta_lstm\ngamma_lstm\ndefault_gru\nbeta_gru\nomicron_lstm")
model_decision = input("Which model do you want to use: ")
if model_decision == "default_lstm":
    model = models.DefaultLSTMModel(X, y, v_X, v_y)
elif model_decision == "beta_lstm":
    model = models.BetaLSTMModel(X, y, v_X, v_y)
elif model_decision == "gamma_lstm":
    model = models.GammaLSTMModel(X, y, v_X, v_y)
elif model_decision == "default_gru":
    model = models.DefaultGRUModel(X, y, v_X, v_y)
elif model_decision == "beta_gru":
    model = models.BetaGRUModel(X, y, v_X, v_y)
elif model_decision == "omicron_lstm":
    dp = DataProcessor("ac_dc.csv", None)
    X, y = dp.training_data("ints", True)
    model = models.OmicronLSTM(X, y, dp, v_X, v_y)
else:
    raise ValueError("No such model exists")

generator = SongGenerator(dp, model)

decision = input("Load or train or load and train model: ")
if decision == "load":
    weights_list = os.listdir("trained_models")

    print("Weights saved are: ")
    for i, name in enumerate(weights_list):
        print(f"{i}: {name}")

    idx_to_load = int(input("Which one to load: "))
    if idx_to_load < 0 or idx_to_load > len(weights_list):
        raise ValueError("Wrong weights choice")

    print(f"Loading {idx_to_load}")

    weights_to_load = weights_list[idx_to_load]
    weights_path = os.path.join("trained_models", weights_to_load)
    model.keras_model.load_weights(weights_path)

    print("Loaded weights :)")

    should_run = True
    while should_run:
        should_generate = input("Do you want to generate lyrics: ")
        if should_generate == "yes":
            temp = input("Provide a temperature: ")
            print(generator.generate(6, 4, float(temp)))
        elif should_generate == "no":
            should_run = False
            print("Goodbye :>")
        else:
            print("I do not understand this input... try again")
elif decision == "train":
    model.train(generator)
    print("Model trained :>")
elif decision == "load and train":
    weights_list = os.listdir("trained_models")

    print("Weights saved are: ")
    for i, name in enumerate(weights_list):
        print(f"{i}: {name}")

    idx_to_load = int(input("Which one to load: "))
    if idx_to_load < 0 or idx_to_load > len(weights_list):
        raise ValueError("Wrong weights choice")

    print(f"Loading {idx_to_load}")

    weights_to_load = weights_list[idx_to_load]
    weights_path = os.path.join("trained_models", weights_to_load)
    model.keras_model.load_weights(weights_path)

    print("Loaded weights :) Training...")

    model.train(generator)

    print("Model trained :>")
else:
    raise ValueError("Unknown decision")
