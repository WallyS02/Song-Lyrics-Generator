import threading
import queue
import requests


def check_proxies(q):
    while not q.empty():
        proxy = q.get()
        try:
            res = requests.get("http://ipinfo.io/json", proxies={"http": proxy, "https": proxy})
        except:
            continue
        if res.status_code == 200:
            with open("valid_proxy_list.txt", "a") as f:
                f.write(proxy + '\n')


def proxies_validation():
    q = queue.Queue()

    with open("proxy_list.txt", "r") as f:
        proxies = f.read().split("\n")
        for p in proxies:
            q.put(p)
    threads = []
    for t in range(10):
        threads.append(threading.Thread(target=check_proxies, args=(q,)))
    for t in range(10):
        threads[t].start()
    for t in range(10):
        threads[t].join()
    with open("valid_proxy_list.txt", "r") as f:
        proxies_list = []
        lines = f.readlines()
        for line in lines:
            proxies_list.append(line)
    return proxies_list
