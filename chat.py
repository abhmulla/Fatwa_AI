import json
import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from unsloth import FastLanguageModel


#############################
# Parameters & Configuration
#############################

# Path constants (update as needed)
JSON_PATH = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\inheritance.json"
FAISS_INDEX_PATH = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\faiss_index.index"

# Model parameters for generation (using your chosen model)
BASE_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
PEFT_CHECKPOINT = "Omartificial-Intelligence-Space/Arabic-DeepSeek-R1-Distill-8B"

# Prompt Template (customize as needed)
PROMPT_TEMPLATE = """
المراجع المتاحة:
{context}

السؤال: {question}

الإجابة:
"""


#############################
# Step 1: Load and Chunk JSON into Documents
#############################
def load_and_chunk_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If data is a dictionary, merge the lists from "inheritance_data" and "math_data"
    if isinstance(data, dict):
        combined = []
        if "inheritance_data" in data:
            combined.extend(data["inheritance_data"])
        if "math_data" in data:
            combined.extend(data["math_data"])
        data = combined  # Now data is a list of entries

    chunks = []
    for i, entry in enumerate(data):
        # Ensure entry is a dict
        if not isinstance(entry, dict):
            print(f"Skipping entry {i} as it is not a dictionary: {entry}")
            continue

        # Use "concept" if available; otherwise, fall back to "heir"
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


#############################
# Step 2: Embed the Documents
#############################
def embed_chunks(chunks: List[Dict], model_name='intfloat/multilingual-e5-base'):
    model = SentenceTransformer(model_name)
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts


#############################
# Step 3: Store Embeddings in FAISS
#############################
def store_in_faiss(embeddings: np.ndarray, index_path=FAISS_INDEX_PATH):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    return index


#############################
# Step 4: Retrieval Function
#############################
def load_faiss_index(index_path=FAISS_INDEX_PATH):
    return faiss.read_index(index_path)

def retrieve(query: str, index, texts: List[str], model, top_k=3) -> List[str]:
    query_embedding = model.encode([query])[0]
    D, I = index.search(np.array([query_embedding]), top_k)
    return [texts[i] for i in I[0]]


#############################
# Step 5: Integrated Model Inference via RAG
#############################
def load_model(model_name="Omartificial-Intelligence-Space/Arabic-DeepSeek-R1-Distill-8B"):
    """Load model and tokenizer using Unsloth for CPU usage."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,             # Keep None to allow CPU fallback
        load_in_4bit=False      # Set False to avoid bitsandbytes (GPU-only)
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def get_response(prompt: str, model, tokenizer, device, max_new_tokens=256):
    # Adjust max input length as needed
    model_max_length = 1024  # Set according to your model's max capacity
    input_max_length = model_max_length - max_new_tokens

    tokenized_input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=input_max_length,
        padding="longest"
    )
    input_ids = tokenized_input.input_ids.to(device)
    attention_mask = tokenized_input.attention_mask.to(device)

    generate_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        do_sample=False  # Greedy decoding
    )
    response = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    # Optionally, post-process the response (e.g. by removing the prompt from it)
    return response.strip()


#############################
# Main Integration Pipeline
#############################
if __name__ == "__main__":
    # Step 1: Load & Chunk JSON
    print("Loading and chunking JSON...")
    chunks = load_and_chunk_json(JSON_PATH)
    if not chunks:
        raise ValueError("No data found. Please check your JSON file.")

    # Step 2: Embed Chunks
    print("Embedding chunks...")
    embeddings, texts = embed_chunks(chunks)

    # Step 3: Store Embeddings in FAISS
    print("Storing embeddings in FAISS...")
    store_in_faiss(np.array(embeddings), index_path=FAISS_INDEX_PATH)

    # Step 4: Retrieve Context for a Query
    user_query = "كيف يتم توزيع الميراث إذا ترك المتوفى ابنة واحدة؟"
    print("\nRetrieving relevant context for the query...")
    # Re-load SentenceTransformer model for retrieval (it can be reused)
    retrieval_model = SentenceTransformer('intfloat/multilingual-e5-base')
    faiss_index = load_faiss_index(index_path=FAISS_INDEX_PATH)
    retrieved_texts = retrieve(user_query, faiss_index, texts, retrieval_model, top_k=3)

    # Build the retrieval context for the prompt
    context = "\n\n".join(f"[مرجع {i+1}]\n{text}" for i, text in enumerate(retrieved_texts))
    print("\nRetrieved Context:\n", context)

    # Build the full prompt from retrieved context and user query
    full_prompt = PROMPT_TEMPLATE.format(context=context, question=user_query)
    print("\nFull Prompt:\n", full_prompt)

    # Step 5: Load Generation Model & Tokenizer using PEFT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model()
    model.to(device)

    # Step 6: Get Answer from Model
    print("\nGenerating answer...")
    final_answer = get_response(full_prompt, model, tokenizer, device, max_new_tokens=256)
    print("\n== Final Answer ==\n", final_answer)
