import random


def create_markov_model(dataset, n_gram=2):
    markov_model = {}
    for i in range(len(dataset) - n_gram - 1):
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
    return markov_model


def generate_lyrics(markov_model, start, limit=100):
    n = 0
    current_state = start
    lyrics = ""
    lyrics += current_state + " "
    while n < limit:
        next_state = random.choices(list(markov_model[current_state].keys()),
                                    list(markov_model[current_state].values()))
        current_state = next_state[0]
        lyrics += current_state + " "
        n += 1
    return lyrics
