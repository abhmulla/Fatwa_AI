import pdfplumber

file = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\Noor.pdf"
pdf = pdfplumber.open(file)
page = pdf.pages[50]
text = page.extract_text()
print(text)

# pdf_path = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\Noor.pdf"
# text = extract_text(pdf_path)

# with open("C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\extracted_text.txt", "w", encoding="utf-8") as f:
#     f.write(text)

