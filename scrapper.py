import random
import pandas as pd
from bs4 import BeautifulSoup
import requests
import os
import time
from ScrapThread import ScrapThread
from proxy_handling import proxies_validation
from main import path


def connect(url, proxies_list):
    headers = {
        'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/534.30 (KHTML, like Gecko) "
                      "Chrome/12.0.742.112 Safari/534.30"}
    main_page = None
    while True:
        try:
            main_page = requests.get(url, headers=headers, proxies={'http': random.choice(proxies_list),
                                                                    'https': random.choice(proxies_list)}, timeout=5.0)
            break
        except:
            continue
    if main_page.status_code != 200:
        raise Exception("Access Denied!")
    return main_page


def get_lyricsAzlyrics(url, proxies_list):
    main_page_text = BeautifulSoup(connect(url, proxies_list).text, "html.parser")
    lyrics = str()
    for goods in main_page_text.find_all("div", {"class": None}):
        if len(goods.text) == 0:
            pass
        lyrics = lyrics + goods.text
    return lyrics


def scrapAzlyrics(url, selected_albums, time_stamp, proxies_list):
    main_page_text = BeautifulSoup(connect(url, proxies_list).text, "html.parser")
    albums = main_page_text.find_all(class_="album")
    if not albums:
        raise Exception("Access Denied!")
    scraped_lyrics = pd.DataFrame(columns=['Title', 'Lyrics'])
    for alb in albums:
        album_title = alb.find("b").text
        album_title = album_title[1:-1]
        print("Checking album: ", album_title)
        if album_title in selected_albums:
            next_song = alb.find_next(class_="listalbum-item")
            if next_song.find(class_="comment"):
                next_song = next_song.find_next(class_=["listalbum-item", "album"])
            while True:
                if not next_song.find(class_="comment"):
                    title = next_song.find("a").text
                    lyrics = get_lyricsAzlyrics(("https://www.azlyrics.com/" + next_song.find("a")['href']),
                                                proxies_list)
                    df = pd.DataFrame({
                        "Title": [title],
                        "Lyrics": [lyrics]
                    })
                    scraped_lyrics = pd.concat([scraped_lyrics, df], ignore_index=True)
                    print("Downloaded song: ", title)
                    next_song = next_song.find_next(class_=["listalbum-item", "album"])
                    if next_song['class'] != ['listalbum-item']:
                        break
                else:
                    next_song = next_song.find_next(class_=["listalbum-item", "album"])
                    if next_song['class'] != ['listalbum-item']:
                        break
                time.sleep(time_stamp)
    return scraped_lyrics


def get_lyricsTekstowo(url, proxies_list):
    main_page_text = BeautifulSoup(connect(url, proxies_list).text, "html.parser")
    lyrics = str()
    for goods in main_page_text.find_all("div", {"class": "inner-text"}):
        if len(goods.text) == 0:
            pass
        if goods.parent.attrs['class'] == ['song-text']:
            lyrics = lyrics + goods.text
    return lyrics


def scrapTekstowo(url, proxies_list):
    main_page_text = BeautifulSoup(connect(url, proxies_list).text, "html.parser")
    song_list = main_page_text.find(class_='ranking-lista')
    songs = song_list.find_all(class_="box-przeboje")
    scraped_lyrics = pd.DataFrame(columns=['Title', 'Lyrics'])
    if not songs:
        raise Exception("Access Denied!")
    for song in songs:
        ref = song.find("a")
        title = ref.text.split('-')[1][1:]
        lyrics = get_lyricsTekstowo('https://www.tekstowo.pl' + ref['href'], proxies_list)
        df = pd.DataFrame({
            "Title": [title],
            "Lyrics": [lyrics]
        })
        scraped_lyrics = pd.concat([scraped_lyrics, df], ignore_index=True)
        print("Downloaded song: ", title)
    return scraped_lyrics


def check_pagination(url, proxies_list):
    main_page_text = BeautifulSoup(connect(url, proxies_list).text, "html.parser")
    pages = main_page_text.find_all(class_='page-link')
    page_number = 0
    for page in pages:
        if page.text != 'Następna >>' and page.text != '<< Poprzednia':
            page_number = page_number + 1
    return page_number / 2


def do_threading(url, selected_albums, time_stamp, proxies_list):
    threads = []
    thread_number = 0
    if url.split('/')[2] == 'www.azlyrics.com':
        thread_number = len(selected_albums)
    if url.split('/')[2] == 'www.tekstowo.pl':
        thread_number = int(check_pagination(url, proxies_list))
    df = pd.DataFrame(columns=['Title', 'Lyrics'])
    for i in range(thread_number):
        t = None
        if url.split('/')[2] == 'www.azlyrics.com':
            t = ScrapThread(target=scrapAzlyrics, args=(url, [selected_albums[i]], time_stamp, proxies_list))
        if url.split('/')[2] == 'www.tekstowo.pl':
            newUrl = url[:-5] + ',strona,' + str(i + 1) + '.html'
            t = ScrapThread(target=scrapTekstowo, args=(newUrl, proxies_list))
        t.daemon = True
        threads.append(t)
    for i in range(thread_number):
        threads[i].start()
    for i in range(thread_number):
        df = pd.concat([df, threads[i].join()], ignore_index=True)
    return df


def scrap_data(url, selected_albums, time_stamp):
    proxies_list = proxies_validation()
    df = do_threading(url, selected_albums, time_stamp, proxies_list)
    if url.split('/')[2] == 'www.azlyrics.com':
        filename = url.split('/')[4][:-5]
        df.to_csv((path + filename))
    if url.split('/')[2] == 'www.tekstowo.pl':
        filename = url.split(',')[1][:-5]
        df.to_csv((path + filename))
    os.remove("valid_proxy_list")
