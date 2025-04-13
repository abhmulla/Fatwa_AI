import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -------------------------
# Configuration
# -------------------------
# Path to your combined JSON file containing Quran, Hadith, and inheritance data.
COMBINED_JSON_PATH = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\inheritance_restructured.json"
TOP_K = 3

# -------------------------
# Document Loader & Formatter
# -------------------------
def load_documents(path: str) -> list:
    """
    Loads the combined JSON file and converts each entry into a flat text string.
    The conversion uses:
      - For "القرآن الكريم": keys: type, concept, text, source.
      - For "حديث": keys: type, concept, text.
      - For "حالة ميراث": keys: type, concept, heir, rule, condition.
    Each document is prefixed with a reference tag.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    for idx, entry in enumerate(data):
        ref = f"[مرجع {idx+1}]"
        doc_text = ""
        data_type = entry.get("type", "")
        
        if data_type == "القرآن الكريم":
            # Ensure required keys exist
            concept = entry.get("concept", "غير محدد")
            text = entry.get("text", "")
            source = entry.get("source", "")
            doc_text = (f"{ref}\n"
                        f"[{data_type}]\n"
                        f"الموضوع: {concept}\n"
                        f"النص: {text}\n"
                        f"المصدر: {source}")
        elif data_type == "حديث":
            concept = entry.get("concept", "غير محدد")
            text = entry.get("text", "")
            doc_text = (f"{ref}\n"
                        f"[{data_type}]\n"
                        f"الموضوع: {concept}\n"
                        f"النص: {text}")
        elif data_type == "حالة ميراث":
            concept = entry.get("concept", "غير محدد")
            heir = entry.get("heir", "غير محدد")
            rule = entry.get("rule", "")
            condition = entry.get("condition", "")
            doc_text = (f"{ref}\n"
                        f"[{data_type}]\n"
                        f"الوريث: {heir}\n"
                        f"الموضوع: {concept}\n"
                        f"القاعدة: {rule}\n"
                        f"الشرط: {condition}")
        else:
            # Fallback for any unrecognized type:
            doc_text = f"{ref}\n{json.dumps(entry, ensure_ascii=False)}"
        
        documents.append(doc_text)
    return documents

# -------------------------
# Build FAISS Index
# -------------------------
def build_faiss_index(documents: list, embedding_model_name='intfloat/multilingual-e5-base'):
    """
    Creates embeddings from the documents and builds a FAISS index.
    """
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(documents, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, model, documents

# -------------------------
# Retrieval Function
# -------------------------
def retrieve_documents(query: str, index, model, documents: list, top_k=TOP_K) -> list:
    """
    Retrieves the top_k most relevant documents for the query.
    """
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [documents[i] for i in indices[0]]

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    # Load the documents
    print("Loading documents from JSON...")
    docs = load_documents(COMBINED_JSON_PATH)
    print(f"Total documents loaded: {len(docs)}")

    # Build the FAISS index using the embedding model
    print("Building FAISS index...")
    index, embed_model, docs = build_faiss_index(docs)
    print("FAISS index built successfully.")

    # Define a sample query for testing retrieval
    query = "ما هو حكم النفقة والوراثة في حال ترك المتوفى ابنة واحدة؟"
    print("\nQuery:")
    print(query)
    
    # Retrieve top relevant documents
    retrieved = retrieve_documents(query, index, embed_model, docs)
    
    # Print the retrieved documents along with their reference tags
    print("\nRetrieved Documents:")
    for idx, doc in enumerate(retrieved):
        print(f"\n--- Document {idx+1} ---")
        print(doc)
