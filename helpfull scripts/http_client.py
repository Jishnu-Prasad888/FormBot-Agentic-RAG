import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from pathlib import Path

BASE_URL = "http://10.64.26.89:8000/"
OUTPUT_DIR = "downloaded"

Path(OUTPUT_DIR).mkdir(exist_ok=True)

html = requests.get(BASE_URL).text
soup = BeautifulSoup(html, "html.parser")

for a in soup.find_all("a"):
    href = a.get("href")

    if href.endswith(".txt"):
        url = urljoin(BASE_URL, href)

        print("Downloading", href)

        r = requests.get(url)

        with open(Path(OUTPUT_DIR) / href, "wb") as f:
            f.write(r.content)