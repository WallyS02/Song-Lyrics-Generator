import os
import random
from scrapper import scrap_data
from markov_model import clean_data
from markov_model import create_markov_model
from markov_model import generate_lyrics

black_sabbath_selected_albums = ["Black Sabbath", "Paranoid", "Master Of Reality", "Vol 4", "Sabbath Bloody Sabbath",
                                 "Sabotage", "Technical Ecstasy", "Never Say Die!", "Heaven And Hell", "Mob Rules",
                                 "Born Again", "Seventh Star", "The Eternal Idol", "Headless Cross", "Tyr",
                                 "Dehumanizer", "Cross Purposes", "Forbidden", "13"]

pink_floyd_selected_albums = ["The Piper At The Gates Of Dawn", "A Saucerful Of Secrets", "Meddle", "More", "Ummagumma",
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
    words_in_verses = int(int(input("Select number of words in verses: ")) / n_gram)
    model = create_markov_model(dataset, n_gram)
    print('\n')
    for i in range(number_of_verses):
        generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), words_in_verses)
        print(generated_lyrics)


def main():
    print("Select data set to use in generation or other option:\n1. Pink Floyd lyrics generation\n2. Black Sabbath "
          "lyrics generation\n3. Bracia Figo Fagot\n4. Paktofonika\n5. Fused English (aka Pink Sabbath) lyrics "
          "generation\n6. Fused Polish (aka Braciofonika Pigo Pagot)\n7. Scrap data\n8. Exit")
    while True:
        selection = int(input())
        match selection:
            case 1:
                generate_song("Pink Floyd.csv")
            case 2:
                generate_song("Black Sabbath.csv")
            case 3:
                generate_song("Bracia Figo Fagot.csv")
            case 4:
                generate_song("Paktofonika.csv")
            case 5:
                generate_song("Pink Sabbath.csv")
            case 6:
                generate_song("Braciofonika Pigo Pagot.csv")
            case 7:
                scrap_data(pink_floyd_selected_albums, black_sabbath_selected_albums, time_stamp)
            case 8:
                break
        print("\nCommand executed")


if __name__ == '__main__':
    main()
