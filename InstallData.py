import csv
import os
import requests
from urllib.parse import urlparse
import re

csv_file = 'all_image_urls.csv'

# The column name or index where image URLs are stored
url_column = 0  # OR set to an integer index like 0 if no headers
# Destination path
save_path = "./images"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

def extract_id_from_url(url):
    """Extract a unique ID from the image URL."""
    # Example: gets '12345' from '.../12345.jpg'
    filename = os.path.basename(urlparse(url).path)
    match = re.match(r'([a-zA-Z0-9]+)', filename)
    return match.group(1) if match else filename

def download_and_save_image(url, save_dir):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad status codes

        image_id = extract_id_from_url(url)
        extension = os.path.splitext(url)[1] or ".jpg"  # Fallback extension
        file_path = os.path.join(save_dir, f"{image_id}{extension}")

        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {url} → {file_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Download each image
with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        for value in row.values():
            url = str(value).strip()
            print(url)
            if url.startswith('http'):
                download_and_save_image(url, save_path)