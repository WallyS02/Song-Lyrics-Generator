import os
import random
from scrapper import scrap_data
from scrapper import clean_data
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


def main():
    print("Select data set to use in generation or other option:\n1. Pink Floyd lyrics generation\n2. Black Sabbath "
          "lyrics generation\n3. Bracia Figo Fagot\n4. Paktofonika\n5. Fused English (aka Pink Sabbath) lyrics "
          "generation\n6. Fused Polish (aka Braciofonika Pigo Pagot)\n7. Prepare data\n8. Scrap data\n9. Exit")
    pink_floyd_dataset = None
    black_sabbath_dataset = None
    pink_sabbath_dataset = None
    paktofonika_dataset = None
    bracia_figo_fagot_dataset = None
    braciofonika_pigo_pagot_dataset = None
    while True:
        selection = int(input())
        match selection:
            case 1:
                model = create_markov_model(pink_floyd_dataset)
                for i in range(5):
                    generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), 10)
                    print(generated_lyrics)
            case 2:
                model = create_markov_model(black_sabbath_dataset)
                for i in range(5):
                    generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), 10)
                    print(generated_lyrics)
            case 3:
                model = create_markov_model(bracia_figo_fagot_dataset)
                for i in range(5):
                    generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), 10)
                    print(generated_lyrics)
            case 4:
                model = create_markov_model(paktofonika_dataset)
                for i in range(5):
                    generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), 10)
                    print(generated_lyrics)
            case 5:
                model = create_markov_model(pink_sabbath_dataset)
                for i in range(5):
                    generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), 10)
                    print(generated_lyrics)
            case 6:
                model = create_markov_model(braciofonika_pigo_pagot_dataset)
                for i in range(5):
                    generated_lyrics = generate_lyrics(model, random.choice(list(model.keys())), 10)
                    print(generated_lyrics)
            case 7:
                path = os.path.dirname(os.path.abspath(__file__))
                path = os.path.join(path, "Data")
                pink_floyd_dataset = clean_data(os.path.join(path, "Pink Floyd.csv"))
                black_sabbath_dataset = clean_data(os.path.join(path, "Black Sabbath.csv"))
                pink_sabbath_dataset = clean_data(os.path.join(path, "Pink Sabbath.csv"))
                paktofonika_dataset = clean_data(os.path.join(path, "Paktofonika.csv"))
                bracia_figo_fagot_dataset = clean_data(os.path.join(path, "Bracia Figo Fagot.csv"))
                braciofonika_pigo_pagot_dataset = clean_data(os.path.join(path, "Braciofonika Pigo Pagot.csv"))
            case 8:
                scrap_data(pink_floyd_selected_albums, black_sabbath_selected_albums, time_stamp)
            case 9:
                break
        print("\nCommand executed")


if __name__ == '__main__':
    main()
