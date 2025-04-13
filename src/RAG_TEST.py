import json
import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


### Step 1: Load and Chunk JSON into Documents
def load_and_chunk_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If data is a dictionary, merge the lists if keys like "inheritance_data" and "math_data" exist.
    if isinstance(data, dict):
        combined = []
        if "inheritance_data" in data:
            combined.extend(data["inheritance_data"])
        if "math_data" in data:
            combined.extend(data["math_data"])
        data = combined  # Now data is a list of entries.

    chunks = []
    for i, entry in enumerate(data):
        # Ensure entry is a dict
        if not isinstance(entry, dict):
            print(f"Skipping entry {i} as it is not a dictionary: {entry}")
            continue

        # Use "concept" if it exists; otherwise, use "heir"
        concept = entry.get("concept", entry.get("heir", f"Unnamed_Concept_{i}"))
        text = f"Concept: {concept}\n"

        if 'definition' in entry:
            text += f"Definition: {entry['definition']}\n"

        if 'goal' in entry:
            text += f"Goal: {entry['goal']}\n"

        if 'method' in entry:
            if isinstance(entry['method'], dict):
                text += "Method:\n" + "\n".join(f"- {k}: {v}" for k, v in entry['method'].items()) + "\n"
            else:
                text += f"Method: {entry['method']}\n"

        if 'rules' in entry:
            if isinstance(entry['rules'], list):
                rule_texts = []
                for rule in entry['rules']:
                    if isinstance(rule, dict):
                        rule_texts.append(", ".join(f"{k}: {v}" for k, v in rule.items()))
                    else:
                        rule_texts.append(str(rule))
                text += "Rules:\n" + "\n".join(f"- {rt}" for rt in rule_texts) + "\n"
            else:
                text += f"Rules: {entry['rules']}\n"

        if 'notes' in entry:
            text += "Notes:\n" + "\n".join(f"- {note}" for note in entry['notes']) + "\n"

        if 'steps' in entry:
            text += "Steps:\n" + "\n".join(f"- {step}" for step in entry['steps']) + "\n"

        if 'example' in entry:
            text += f"Example:\n{json.dumps(entry['example'], ensure_ascii=False)}\n"

        if 'examples' in entry:
            for ex in entry['examples']:
                text += f"Example:\n{json.dumps(ex, ensure_ascii=False)}\n"

        chunks.append({
            "id": i,
            "concept": concept,
            "text": text.strip()
        })

    return chunks


### Step 2: Embed the Documents
def embed_chunks(chunks: List[Dict], model_name='intfloat/multilingual-e5-base'):
    model = SentenceTransformer(model_name)
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts


### Step 3: Store Embeddings in FAISS
def store_in_faiss(embeddings: np.ndarray, index_path="C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\faiss_index.index"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    return index


### Step 4: Retrieval Function
def load_faiss_index(index_path="C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\faiss_index.index"):
    return faiss.read_index(index_path)

def retrieve(query: str, index, texts: List[str], model, top_k=3) -> List[str]:
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    return [texts[i] for i in I[0]]


### Run the Pipeline
if __name__ == "__main__":
    # Path to your structured JSON file
    json_path = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\inheritance.json"  # Make sure this path is correct

    # Load and chunk
    print("Loading and chunking JSON...")
    chunks = load_and_chunk_json(json_path)

    if not chunks:
        raise ValueError("No data found. Please check your JSON file.")

    # Embed
    print("Embedding chunks...")
    embeddings, texts = embed_chunks(chunks)

    # Store in FAISS
    print("Storing in FAISS...")
    index = store_in_faiss(np.array(embeddings))

    # Test retrieval
    print("\nReady for testing.")
    user_query = "كيف يتم توزيع الميراث إذا ترك المتوفى ابنة واحدة"
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    index = load_faiss_index()
    results = retrieve(user_query, index, texts, model)

    print("\nTop relevant chunks:")
    for i, r in enumerate(results):
        print(f"\n--- Chunk {i+1} ---\n{r}")
