import copy
import math
import random
import re
from nltk import SyllableTokenizer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def clean_data(name):
    document = pd.read_csv(name, usecols=["Lyrics"])
    rows = document["Lyrics"].values.tolist()
    dataset = []
    for lyric in rows:
        if isinstance(lyric, str):
            lyric = lyric.lower()
            lyric = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]", "", lyric)
            lyric = re.sub(r"\([A-Za-z0-9:\s\.\?\,\&\*]+\)", "", lyric)
            lyric = re.sub(r"\[[A-Za-z0-9:\s\.\?\,\&\*]+\]", "", lyric)
            lyric = re.sub(r"[A-Za-z0-9]+::", "", lyric)
            lyric = re.sub(r"[A-Za-z0-9]+:", "", lyric)
            lyric = re.sub(r"/[A-Za-z0-9]+", "", lyric)
            lyric = re.sub(r"x[0-9]", "", lyric)
            forbidden_words = ['chorus', 'refrain', 'coda', 'solo', 'intro', 'introduction', 'verse', 'pre-chorus',
                               'post-chorus', 'bridge', 'outro', 'ref']
            tokens = word_tokenize(lyric)
            words = [word for word in tokens if word.isalpha()]
            words = [word for word in words if word not in forbidden_words]
            dataset += words
    print(name.split('\\')[-1], "number of words in cleaned data: ", len(dataset))
    return dataset


def create_markov_model(dataset, n_gram):
    markov_model = {}
    for i in range(len(dataset) - n_gram):
        current_state, next_state = "", ""
        for j in range(n_gram):
            current_state += dataset[i + j] + " "
        next_state += dataset[i + n_gram]
        current_state = current_state[:-1]
        if current_state not in markov_model:
            markov_model[current_state] = {}
            markov_model[current_state][next_state] = 1
        else:
            if next_state in markov_model[current_state]:
                markov_model[current_state][next_state] += 1
            else:
                markov_model[current_state][next_state] = 1
    for current_state, transition in markov_model.items():
        total = sum(transition.values())
        for state, count in transition.items():
            markov_model[current_state][state] = count / total
    return markov_model


def default_next_state(markov_model, current_state, lyrics):
    next_state = random.choices(list(markov_model[current_state].keys()),
                                list(markov_model[current_state].values()))
    lyrics += next_state[0] + " "
    n_gram = len(current_state.split(" "))
    current_state = ""
    for i in range(n_gram + 1, 1, -1):
        current_state += lyrics.split(" ")[-i] + " "
    current_state = current_state[:-1]
    return current_state, lyrics


def rhyming_next_state(rime_states, current_state, lyrics):
    next_state = random.choices(list(rime_states.keys()),
                                list(rime_states.values()))
    lyrics += next_state[0] + " "
    n_gram = len(current_state.split(" "))
    current_state = ""
    for i in range(n_gram + 1, 1, -1):
        current_state += lyrics.split(" ")[-i] + " "
    current_state = current_state[:-1]
    return current_state, lyrics


def generate_lyrics(markov_model, start, limit, try_rhyme, rime):
    n = 0
    current_state = start
    lyrics = ""
    lyrics += current_state + " "
    lyrics = lyrics[0].upper() + lyrics[1:]
    while n < limit:
        if n == limit - 1 and try_rhyme is True:
            rime = rime.split(" ")[-1]
            tk = SyllableTokenizer()
            rime_syllab = tk.tokenize(rime)[-1]
            rime_states = {}
            for state, probability in markov_model[current_state].items():
                syllab = tk.tokenize(state)[-1]
                if rime_syllab == syllab and rime != state:
                    rime_states.update({state: probability})
            if rime_states:
                current_state, lyrics = rhyming_next_state(rime_states, current_state, lyrics)
            else:
                current_state, lyrics = default_next_state(markov_model, current_state, lyrics)
        else:
            current_state, lyrics = default_next_state(markov_model, current_state, lyrics)
        n += 1
    return lyrics, current_state


def get_bleu(sentence, remaining_sentences):
    lst = []
    smoothie = SmoothingFunction()
    for i in remaining_sentences:
        bleu = sentence_bleu(sentence, i, smoothing_function=smoothie.method1)
        lst.append(bleu)
    return lst


def self_BLEU(sentences):
    bleu_scores = []
    for i in sentences:
        sentences_copy = copy.deepcopy(sentences)
        sentences_copy.remove(i)
        bleu = get_bleu(i, sentences_copy)
        bleu_scores.append(bleu)
    return np.mean(bleu_scores)


def zipfs_law(dataset, name, firstValues=1000):
    histogram = {}
    for state in dataset:
        if state in histogram.keys():
            histogram[state] += 1
        else:
            histogram[state] = 1
    keys = list(histogram.keys())
    values = list(histogram.values())
    sorted_value_index = np.argsort(-np.array(values))
    sorted_histogram = {keys[i]: values[i] for i in sorted_value_index}
    plt.bar([i for i in range(min(len(sorted_histogram), firstValues))],
            [list(sorted_histogram.values())[i] for i in range(min(len(sorted_histogram), firstValues))])
    plt.xlabel("states")
    plt.ylabel("occurrences")
    plt.title(name + " state histogram")
    plt.tight_layout()
    plt.show()
    constant_list = []
    for i, state in enumerate(sorted_histogram.values()):
        if i == min(len(sorted_histogram), firstValues):
            break
        constant_list.append((i + 1) * state)
    plt.xlabel("states")
    plt.ylabel("constants")
    plt.title(name + " state constants plot")
    plt.tight_layout()
    plt.bar([i for i in range(min(len(sorted_histogram), firstValues))], constant_list)
    plt.show()


def heaps_law(dataset, n_gram):
    unique_states = []
    for state in dataset:
        if state not in unique_states:
            unique_states.append(state)
    return int(math.factorial(len(unique_states)) / math.factorial(len(unique_states) - n_gram)), len(dataset) ** n_gram


def plot_heaps_laws(datasets, n_grams):
    for n_gram in n_grams:
        x = []
        y = []
        for dataset in datasets:
            unique, total = heaps_law(dataset, n_gram)
            x.append(total)
            y.append(unique)
        plt.plot(x, y, linewidth=1.0)
        plt.xlabel("total number of states")
        plt.ylabel("unique number of states")
        plt.title("Heap's law")
        plt.legend(["n_gram: " + str(n_gram)])
        plt.tight_layout()
        plt.show()


def cross_entropy(model, text, k):
    counts = {}
    for i in range(len(text) - k):
        gram = ""
        for j in range(k):
            gram += text[i + j] + " "
        gram = gram[:-1]
        if gram not in counts:
            counts[gram] = 0
        counts[gram] += 1

    total = sum(counts.values())
    probs = {gram: count / total for gram, count in counts.items()}

    entropy = 0
    for i in range(len(text) - k):
        gram = ""
        for j in range(k):
            gram += text[i + j] + " "
        gram = gram[:-1]
        next_word = text[i + k]
        if gram in model:
            prob = model[gram].get(next_word, 0)
            entropy -= np.log2(prob) * probs[gram]
    return entropy


def perplexity(entropy):
    return pow(2, entropy)
