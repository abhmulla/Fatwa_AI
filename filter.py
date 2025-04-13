import pandas as pd
import re

# File paths
input_file = "C:\\Users\\user\\desktop\\fatwa_ai\\fatwaset\\Baz_after.xlsx"
keywords_file = "C:\\Users\\user\\desktop\\fatwa_ai\\words.txt"
output_file = "C:\\Users\\user\\desktop\\fatwa_ai\\fatwaset_filtered\\Baz_after_filtered.xlsx"
evidence_file = "C:\\Users\\user\\desktop\\fatwa_ai\\fatwaset_filtered\\Baz_after_evidence.xlsx"

# Load data
df = pd.read_excel(input_file)
with open(keywords_file, 'r', encoding='utf-8') as f:
    keywords = [line.strip() for line in f]

# Enhanced filtering function using pandas vectorization
pattern = '|'.join(keywords)
mask = (
    df['Column1.question'].str.contains(pattern, case=False, na=False, regex=True) |
    df['Column1.answer'].str.contains(pattern, case=False, na=False, regex=True) 
    #df['Column1.title'].str.contains(pattern, case=False, na=False, regex=True)
)
filtered_df = df[mask].copy()

# Regex patterns for evidence extraction
quran_pattern = r'\(([^)]+)\)'  # Matches content between parentheses
hadith_pattern = r'«([^»]+)»'   # Matches content between «»

def process_text(text):
    """Extract evidence and clean text with comprehensive regex handling"""
    text = str(text)  # Handle NaN values
    quran_evidence = re.findall(quran_pattern, text)
    hadith_evidence = re.findall(hadith_pattern, text)
    
    # Clean text while preserving original structure
    cleaned_text = re.sub(quran_pattern, '', text)
    cleaned_text = re.sub(hadith_pattern, '', cleaned_text)
    
    # Remove extra spaces created by replacements
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text, quran_evidence + hadith_evidence

# Process all answers
filtered_df[['cleaned_answer', 'evidence']] = filtered_df['Column1.answer'].apply(
    lambda x: pd.Series(process_text(x)))
filtered_df = filtered_df.drop(columns=['Column1.answer']).rename(
    columns={'cleaned_answer': 'Column1.answer'})
# Save evidence separately
evidence_list = []
for idx, row in filtered_df.iterrows():
    for ref in row['evidence']:
        evidence_list.append({
            'original_index': idx,
            'reference': ref,
            'question': row['Column1.question'],
            #'title': row['Column1.title']
        })
        
evidence_df = pd.DataFrame(evidence_list)

# Save files
filtered_df.to_excel(output_file, index=False)
evidence_df.to_excel(evidence_file, index=False)

print(f"""
  Processing complete!
- Filtered fatwas saved to: {output_file}
- Extracted {len(evidence_df)} evidence references saved to: {evidence_file}
- Original answers cleaned with evidence removed
- Evidence context preserved with question/title metadata
""")