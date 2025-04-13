import pandas as pd

# Load filtered data
df = pd.read_excel("C:\\Users\\user\\desktop\\fatwa_ai\\fatwaset_filtered\\Othaimin_after_filtered.xlsx")

# Convert to instruction-response format
formatted_data = []
for _, row in df.iterrows():
    formatted_data.append({
        "instruction": row["Column1.question"],
        "input": "",  # Optional: additional context
        "output": row["Column1.answer"]
    })

# Save as JSONL
import json
with open("C:\\Users\\user\\desktop\\fatwa_ai\\fatwaset_filtered\\Othaimin_train.jsonl", "w", encoding="utf-8") as f:
    for entry in formatted_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Done")