import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def get_imdb_poster(imdb_id):
    url = f"https://www.imdb.com/title/{imdb_id}/"
    try:
        r = session.get(url, timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Request error for {imdb_id}: {e}")
        return None

    soup = BeautifulSoup(r.text, "lxml")

    poster_img = soup.select_one("div.ipc-poster img")
    if poster_img and poster_img.get("src"):
        return poster_img["src"]
    return None


def get_posters_parallel(imdb_ids, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(get_imdb_poster, imdb_ids))
    return results
