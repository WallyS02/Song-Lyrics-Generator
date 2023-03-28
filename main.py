import os
import random
import pandas as pd
from scrapper import scrap_data
from markov_model import clean_data
from markov_model import create_markov_model
from markov_model import generate_lyrics

blacksabbath_selected_albums = ["Black Sabbath", "Paranoid", "Master Of Reality", "Vol 4", "Sabbath Bloody Sabbath",
                                "Sabotage", "Technical Ecstasy", "Never Say Die!", "Heaven And Hell", "Mob Rules",
                                "Born Again", "Seventh Star", "The Eternal Idol", "Headless Cross", "Tyr",
                                "Dehumanizer", "Cross Purposes", "Forbidden", "13"]

pinkfloyd_selected_albums = ["The Piper At The Gates Of Dawn", "A Saucerful Of Secrets", "Meddle", "More", "Ummagumma",
                             "Atom Heart Mother", "Obscured By Clouds", "The Dark Side Of The Moon",
                             "Wish You Were Here", "Animals", "The Wall", "The Final Cut",
                             "A Momentary Lapse Of Reason", "The Division Bell"]

time_stamp = 3.5
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "Data")


def generate_song(name):
    dataset = clean_data(os.path.join(path, name))
    n_gram = int(input("Select number of words in Markov state: "))
    number_of_verses = int(input("Select number of verses: "))
    words_in_verses = int((int(input("Select number of words in verses: ")) - 1) / n_gram)
    # degree_of_chain = int(input("Select degree of chain: "))
    model = create_markov_model(dataset, n_gram)
    print('\n')
    last_state = random.choice(list(model.keys()))
    rime = None
    for i in range(number_of_verses):
        generated_lyrics, last_state = generate_lyrics(model, last_state, words_in_verses, True if i == 0 else False, rime)
        print(generated_lyrics)
        rime = last_state
        last_state = random.choices(list(model[last_state].keys()),
                                    list(model[last_state].values()))[0]


def scraping():
    with open("links.txt", "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i != (len(lines) - 1):
                print(str(i) + ".", lines[i][:-1])
            else:
                print(str(i) + ".", lines[i])
    line_index = int(input("Select url to scrap: "))
    url = lines[line_index]
    if line_index != (len(lines) - 1):
        url = url[:-1]
    if url.split('/')[2] == 'www.azlyrics.com':
        selected_albums_name = url.split('/')[4][:-5] + "_selected_albums"
        if selected_albums_name in globals():
            selected_albums = globals()[selected_albums_name]
            scrap_data(url, selected_albums, time_stamp)
        else:
            print("Define selected albums in global list variable in format: bandname_selected_albums")
            return
    if url.split('/')[2] == 'www.tekstowo.pl':
        scrap_data(url, [], 0.0)


def merging():
    name1 = input("Select first band file: ")
    if os.path.exists(os.path.join(path, name1)):
        df1 = pd.read_csv(os.path.join(path, name1))
    else:
        print("No such file in directory!")
        return
    name2 = input("Select second band file: ")
    if os.path.exists(os.path.join(path, name2)):
        df2 = pd.read_csv(os.path.join(path, name2))
    else:
        print("No such file in directory!")
        return
    dfResult = pd.concat([df1, df2], ignore_index=True)
    result_name = input("Select name of result file: ")
    dfResult.to_csv(os.path.join(path, result_name))


def main():
    print("Select data set to use in generation or other option:\n1. Generate text based on input filename\n2. Scrap "
          "data\n3. Merge CSV band's songs\n4. Exit")
    while True:
        selection = int(input())
        match selection:
            case 1:
                name = input("Select name of data file: ")
                generate_song(name)
            case 2:
                scraping()
            case 3:
                merging()
            case 4:
                break
        print("\nCommand executed")


if __name__ == '__main__':
    main()
