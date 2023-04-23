import os
import random
import pandas as pd
from scrapper import scrap_data
from markov_model import clean_data, create_markov_model, generate_lyrics, self_BLEU, zipfs_law, plot_heaps_laws, cross_entropy, perplexity
import json

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
pathData = os.path.join(path, "Data")
pathModels = os.path.join(path, "Models")


def print_file_list(filepath):
    filelist = []
    for file in os.listdir(filepath):
        if os.path.isfile(os.path.join(filepath, file)):
            filelist.append(file)
    i = 0
    for file in filelist:
        print(i, ": ", file)
        i += 1
    return filelist


def create_model():
    filelist = print_file_list(pathData)
    name = filelist[int(input("Select datafile: "))]
    dataset = clean_data(os.path.join(pathData, name))
    n_gram = int(input("Select number of words in Markov state: "))
    model = create_markov_model(dataset, n_gram)
    model_name = input("Select model name: ")
    with open(os.path.join(pathModels, model_name) + '.json', 'w') as model_file:
        model_file.write(json.dumps(model))


def generate_song():
    filelist = print_file_list(pathModels)
    model_name = filelist[int(input("Select model: "))]
    with open(os.path.join(pathModels, model_name), 'r') as model_file:
        model = json.loads(model_file.read())
    number_of_verses = int(input("Select number of verses: "))
    words_in_verses = int(input("Select number of words in verses: ")) - len(list(model.keys())[0].split(' '))
    print('\n')
    rime = None
    song = []
    for i in range(number_of_verses):
        generated_lyrics, rime = generate_lyrics(model, random.choice(list(model.keys())), words_in_verses, True if i % 2 == 1 else False, rime)
        print(generated_lyrics)
        for state in generated_lyrics.split():
            song.append(state.lower())
    return song


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
    df = pd.DataFrame(columns=['Title', 'Lyrics'])
    print("Select files to merge: ")
    filelist = []
    for file in os.listdir(pathData):
        if os.path.isfile(os.path.join(pathData, file)):
            filelist.append(file)
    while True:
        i = 0
        for file in filelist:
            print(i, ": ", file)
            i += 1
        print(i, ": That's all")
        option = int(input("Select option: "))
        if option == i:
            break
        else:
            df1 = pd.read_csv(os.path.join(pathData, filelist[option]))
            df = pd.concat([df, df1], ignore_index=True)
            filelist.pop(option)
    result_name = input("Select name of result file: ")
    df.to_csv(os.path.join(pathData, result_name))


def main():
    print("Select option:\n1. Create model based on datafile\n2. Generate lyrics with model\n3. Scrap "
          "data\n4. Merge CSV band's songs\n5. Exit")
    while True:
        selection = int(input())
        match selection:
            case 1:
                create_model()
                pass
            case 2:
                generate_song()
                pass
            case 3:
                scraping()
            case 4:
                merging()
            case 5:
                break
        print("\nCommand executed")


if __name__ == '__main__':
    main()
