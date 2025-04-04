import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import csv
import time

url = "https://myislam.org/quran-verses/inheritance/"

options = uc.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
driver = uc.Chrome(options=options)
driver.get(url)
time.sleep(15)  # Wait for Cloudflare challenge

soup = BeautifulSoup(driver.page_source, "html.parser")
verse_containers = soup.find_all("div", class_="dua-container")

verses_data = []

for container in verse_containers:
    try:
        # Reference extraction with safety check
        reference_div = container.find("div", class_="chapter-title")
        reference = reference_div.get_text(strip=True) if reference_div else "No Reference"
        
        # Arabic text extraction with exact class match
        arabic_div = container.find("div", class_="arabic-text ayat-arabic-text")
        arabic_text = arabic_div.get_text(strip=True) if arabic_div else "No Arabic Text"
        
        # Only save entries with both values
        if reference != "No Reference" and arabic_text != "No Arabic Text":
            verses_data.append({
                "reference": reference,
                "arabic_text": arabic_text
            })
            
    except Exception as e:
        print(f"Error processing container: {e}")
        continue

# Save to CSV
with open("C:\\Users\\user\\desktop\\fatwa_ai\\quran_verses.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Reference", "Arabic Text"])
    for verse in verses_data:
        writer.writerow([verse["reference"], verse["arabic_text"]])

print(f"Successfully scraped {len(verses_data)} verses")
driver.quit()