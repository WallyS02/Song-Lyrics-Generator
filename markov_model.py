import random
import re
from nltk import SyllableTokenizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from scipy import sparse


def clean_data(name):
    document = pd.read_csv(name, usecols=["Lyrics"])
    rows = document["Lyrics"].values.tolist()
    dataset = []
    for lyric in rows:
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
    for i in range(len(dataset) - 1 - 2 * n_gram):
        current_state, next_state = "", ""
        for j in range(n_gram):
            current_state += dataset[i + j] + " "
            next_state += dataset[i + j + n_gram] + " "
        current_state = current_state[:-1]
        next_state = next_state[:-1]
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
    """matrix = [[0 for _ in range(len(markov_model.items()))] for _ in range(int(len(markov_model.items())))]
    for current_state, transition in markov_model.items():
        tempRow = list(markov_model.items())
        indexRow = [idx for idx, key in enumerate(tempRow) if key[0] == current_state]
        total = sum(transition.values())
        for state, count in transition.items():
            tempCol = list(transition.items())
            indexCol = [idx for idx, key in enumerate(tempCol) if key[0] == state]
            markov_model[current_state][state] = count / total
            matrix[indexRow[0]][indexCol[0]] = markov_model[current_state][state]
    matrix = np.array(matrix)
    for i in range(n_step):
        matrix = matrix.dot(matrix)
        for current_state, transition in markov_model.items():
            tempRow = list(markov_model.items())
            indexRow = [idx for idx, key in enumerate(tempRow) if key[0] == current_state]
            for state, count in transition.items():
                tempCol = list(transition.items())
                indexCol = [idx for idx, key in enumerate(tempCol) if key[0] == state]
                markov_model[current_state][state] += matrix[indexRow[0]][indexCol[0]]"""
    return markov_model


def generate_lyrics(markov_model, start, limit, isStartingVerse, rime):
    n = 0
    current_state = start
    lyrics = ""
    lyrics += current_state + " "
    lyrics = lyrics[0].upper() + lyrics[1:]
    while n < limit:
        if n == limit - 1 and not isStartingVerse:
            rime = rime.split(" ")[-1]
            tk = SyllableTokenizer()
            rime_syllab = tk.tokenize(rime)[-1]
            rime_states = {}
            for state, probability in markov_model[current_state].items():
                word = state.split(" ")[-1]
                syllab = tk.tokenize(word)[-1]
                if rime_syllab == syllab and rime != word:
                    rime_states.update({state: probability})
            if rime_states:
                next_state = random.choices(list(rime_states.keys()),
                                            list(rime_states.values()))
                current_state = next_state[0]
            else:
                next_state = random.choices(list(markov_model[current_state].keys()),
                                            list(markov_model[current_state].values()))
                current_state = next_state[0]
        else:
            next_state = random.choices(list(markov_model[current_state].keys()),
                                        list(markov_model[current_state].values()))
            current_state = next_state[0]
        lyrics += current_state + " "
        n += 1
    return lyrics, current_state
