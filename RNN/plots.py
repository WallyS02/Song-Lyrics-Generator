import math

import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import numpy as np


def plot_omicron_metrics():
    data = pd.read_csv(join("trained_models", "omicron_lstm.log"))

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f"Model: omicron_lstm")
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.25,
                        hspace=0.8)
    fig.set_figwidth(10)

    accuracy = data["accuracy"]
    ax[0][0].plot(range(1, 250), np.linspace(0, accuracy[0], 249), "deepskyblue", linestyle="--")
    ax[0][0].plot(range(250, len(accuracy) + 250), accuracy, 'deepskyblue')
    ax[0][0].set(xlabel="Numer epoki", ylabel="Dokładność (accuracy)")
    ax[0][0].set_title("Dokładność")

    precision = data["precision"]
    ax[0][1].plot(range(1, 250), np.linspace(0, precision[0], 249), "limegreen", linestyle="--")
    ax[0][1].plot(range(250, len(precision) + 250), precision, 'limegreen')
    ax[0][1].set(xlabel="Numer epoki", ylabel="Precyzja (precision)")
    ax[0][1].set_title("Precyzja")

    recall = data["recall"]
    ax[1][0].plot(range(1, 250), np.linspace(0, recall[0], 249), "orange", linestyle="--")
    ax[1][0].plot(range(250, len(recall) + 250), recall, 'orange')
    ax[1][0].set(xlabel="Numer epoki", ylabel="Zwrot (recall)")
    ax[1][0].set_title("Zwrot")

    loss = data["loss"]
    ax[1][1].plot(range(1, 250), np.linspace(8.06, loss[0], 249), "r", linestyle="--")
    ax[1][1].plot(range(250, len(loss) + 250), loss, 'r')
    ax[1][1].set(xlabel="Numer epoki", ylabel="Koszt (loss)")
    ax[1][1].set_title("Koszt")

    fig.savefig(join("metrics_plots", f"omicron_lstm.png"))


def plot_metrics_for(filename, model_name):
    data = pd.read_csv(join("trained_models", filename))

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(f"Model: {model_name}")
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.25,
                        hspace=0.8)
    fig.set_figwidth(10)

    accuracy = data["accuracy"]
    ax[0][0].plot(range(1, len(accuracy) + 1), accuracy, 'deepskyblue')
    ax[0][0].set(xlabel="Numer epoki", ylabel="Dokładność (accuracy)")
    ax[0][0].set_title("Dokładność")

    precision = data["precision"]
    ax[0][1].plot(range(1, len(precision) + 1), precision, 'limegreen')
    ax[0][1].set(xlabel="Numer epoki", ylabel="Precyzja (precision)")
    ax[0][1].set_title("Precyzja")

    recall = data["recall"]
    ax[1][0].plot(range(1, len(recall) + 1), recall, 'orange')
    ax[1][0].set(xlabel="Numer epoki", ylabel="Zwrot (recall)")
    ax[1][0].set_title("Zwrot")

    loss = data["loss"]
    perplexity = [math.e**l for l in list(loss)]

    ax[1][1].plot(range(1, len(loss) + 1), loss, 'r', label="Koszt")
    pax = ax[1][1].twinx()
    pax.plot(range(1, len(loss) + 1), perplexity, 'm', label="Perpleksja")
    pax.set_ylabel("Perpleksja")
    pax.legend(loc="right")
    ax[1][1].set(xlabel="Numer epoki", ylabel="Koszt")
    ax[1][1].legend(loc="upper right")
    ax[1][1].set_title("Koszt")

    fig.savefig(join("metrics_plots", f"{model_name}.png"))


plot_metrics_for("default_gru.log", "default_gru")
plot_metrics_for("beta_gru.log", "beta_gru")
plot_metrics_for("gamma_lstm.log", "gamma_lstm")
plot_omicron_metrics()
plot_metrics_for("default_lstm.log", "default_lstm")
plot_metrics_for("beta_lstm.log", "beta_lstm")
