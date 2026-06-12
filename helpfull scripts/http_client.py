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

    if not href or not href.endswith(".json"):
        continue

    url = urljoin(BASE_URL, href)

    # Remove leading web_
    filename = href
    if filename.startswith("web_"):
        filename = filename[4:]

    print("Downloading", href, "->", filename)

    r = requests.get(url)

    with open(Path(OUTPUT_DIR) / filename, "wb") as f:
        f.write(r.content)