import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv(dotenv_path="C:/Users/user/Desktop/fatwa_ai/deep.env")

# Configuration
JSON_PATH = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\inheritance_restructured.json"
FAISS_INDEX_PATH = "C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA\\faiss_index.index"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-base'

# Initialize DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

PROMPT_TEMPLATE = """\
أنت عالم متخصص في علم الفرائض والمواريث في الشريعة الإسلامية، وتتمتع بمعرفة عميقة بآيات القرآن الكريم، الأحاديث النبوية، وأحكام المواريث.

أجب عن السؤال التالي بدقة ووضوح، مستخدمًا كلًا من السؤال، وسياقه، والمراجع المقدمة. 
إذا استخدمت آية من القرآن أو حديثًا نبويًا، فاذكر مصدرها بوضوح (اسم السورة ورقم الآية، أو اسم الحديث ورقمه إن وجد).

استخدم لغة علمية عربية واضحة كما يفعل العلماء.
إذا لم تكن هناك معلومات كافية حتى مع السياق والسؤال، قل بصدق: "لا أعلم".

المراجع المتاحة:
{context}

السؤال:
{question}

الإجابة العلمية المدعومة بالمصادر:
"""


def load_and_format_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for idx, entry in enumerate(data):
        ref = f"[مرجع {idx+1}]"
        data_type = entry.get("type", "")
        if data_type == "القرآن الكريم":
            doc_text = (f"{ref}\n[القرآن الكريم]\nالموضوع: {entry.get('concept')}\n"
                        f"النص: {entry.get('text')}\nالمصدر: {entry.get('source')}")
        elif data_type == "حديث":
            doc_text = (f"{ref}\n[حديث]\nالموضوع: {entry.get('concept')}\nالنص: {entry.get('text')}")
        elif data_type == "حالة ميراث":
            doc_text = (f"{ref}\n[حالة ميراث]\nالوريث: {entry.get('heir')}\n"
                        f"الموضوع: {entry.get('concept')}\nالقاعدة: {entry.get('rule')}\n"
                        f"الشرط: {entry.get('condition')}")
        else:
            doc_text = f"{ref}\n{json.dumps(entry, ensure_ascii=False)}"
        documents.append(doc_text)
    return documents

def build_faiss_index(docs, model_name=EMBEDDING_MODEL):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, model, docs


def retrieve_documents(query, index, model, docs, top_k=3):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [docs[i] for i in indices[0]]

def get_deepseek_response(prompt: str) -> str:
    """Get response from DeepSeek API with Islamic law context"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "أنت فقيه إسلامي متخصص في علم المواريث. استخدم المصادر الشرعية في إجاباتك."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=512,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {str(e)}")
        return "عذرًا، حدث خطأ في الحصول على الإجابة."

if __name__ == "__main__":
    
    # Integrated pipeline execution
    documents = load_and_format_documents(JSON_PATH)
    index, model, docs = build_faiss_index(documents)
    query = "ماتت امرأة عن زوج وثلاث بنات ابن وأخ شقیق، كيف يتم توزيع التركة؟"
    retrieved_context = retrieve_documents(query, index, model, docs)
    context = "\n\n".join(retrieved_context)
    final_prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    response = get_deepseek_response(final_prompt)
    print(response)
    