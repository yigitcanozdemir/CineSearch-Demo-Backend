import requests
from bs4 import BeautifulSoup


def get_imdb_poster(imdb_id):
    url = f"https://www.imdb.com/title/{imdb_id}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    poster_div = soup.find("div", class_="ipc-poster")
    if poster_div:
        img_tag = poster_div.find("img")
        if img_tag and img_tag.get("src"):
            return img_tag["src"]

    return None
