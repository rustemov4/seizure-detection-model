#!/usr/bin/env python3
import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse

BASE_URL = "https://physionet.org/files/siena-scalp-eeg/1.0.0/"
OUTPUT_DIR = "siena-scalp-eeg-database-1.0.0"

def download_file(url, local_path):
    print(f"[INFO] Downloading {url} to {local_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[INFO] Download complete: {local_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")

def scrape_and_download(url, base_output):
    print(f"[INFO] Scraping URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to access {url}: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")
    for link in links:
        href = link.get("href")
        if not href or href in ["/", "../"]:
            continue

        absolute_url = urljoin(url, href)
        parsed = urlparse(absolute_url)
        absolute_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

        rel_path = os.path.relpath(parsed.path, urlparse(BASE_URL).path)
        local_path = os.path.join(base_output, rel_path)

        if href.endswith("/"):
            os.makedirs(local_path, exist_ok=True)
            scrape_and_download(absolute_url, base_output)
        else:
            if os.path.exists(local_path):
                print(f"[INFO] File already exists: {local_path}")
            else:
                download_file(absolute_url, local_path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scrape_and_download(BASE_URL, OUTPUT_DIR)

if __name__ == "__main__":
    main()
