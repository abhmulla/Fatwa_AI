import requests
from bs4 import BeautifulSoup
import csv

url = "https://sunnah.com/nasai/44"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Encoding": "gzip, deflate, br",  # Explicitly accept compression
}

response = requests.get(url, headers=headers)
response.encoding = "utf-8"  # Force UTF-8 override
soup = BeautifulSoup(response.content, "html.parser")  # Parse raw content

hadith_data = []

for container in soup.find_all("div", class_="actualHadithContainer"):
    # Extract Arabic text
    arabic_text_div = container.find("div", class_="arabic_hadith_full")
    arabic_text = arabic_text_div.get_text(strip=True) if arabic_text_div else "N/A"

    # Extract reference (now under <table> with class "hadith_reference")
    reference_table = container.find("table", class_="hadith_reference")
    reference = reference_table.get_text(" ", strip=True) if reference_table else "N/A"

    hadith_data.append({
        "arabic_text": arabic_text,
        "reference": reference
    })


csv_filename = "C:\\Users\\user\\desktop\\fatwa_ai\\hadith_data.csv"
with open(csv_filename, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["arabic_text", "reference"])
    for entry in hadith_data:
        writer.writerow([entry["arabic_text"], entry["reference"]])

print(f"Data saved to {csv_filename}. Check the file in Notepad++/VS Code (not Excel).")