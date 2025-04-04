from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ==== Set Up Llama Index ==== 
Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
Settings.llm = None
Settings.chunck_size = 512
Settings.chuck_overlap = 64

documents = SimpleDirectoryReader("C:\\Users\\user\\desktop\\fatwa_ai\\RAGDATA").load_data()
print(len(documents))

index = VectorStoreIndex.from_documents(documents)
top_k = 2

retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
query_engine = RetrieverQueryEngine(
    retriever=retriever, 
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)

# Load LLM
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "aubmindlab/aragpt2-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# ==== Prompt Template ====
prompt_ar = """
\n\nالمراجع المتاحة:
{context}

الإجابة يجب أن:
1. تستخدم فقط المعلومات من المراجع أعلاه
2. تذكر رقم المرجع بين قوسين مثل (المرجع 1)
3. ترفض الإجابة إذا لم يوجد دليل واضح
"""

### Input: [|Human|] {Question}
### Response: [|AI|]"""

#Get response from LLM
def get_response(text, tokenizer=tokenizer, model=model):
    # Calculate max lengths based on model's capabilities
    model_max_length = 1024  # aragpt2-base's max position embeddings
    max_new_tokens = 256
    tokenizer_max_length = model_max_length - max_new_tokens

    # Tokenize with adjusted length limits
    tokenized_input = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=tokenizer_max_length,
        padding="longest"
    )
    input_ids = tokenized_input.input_ids.to(device)
    attention_mask = tokenized_input.attention_mask.to(device)
    
    generate_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        do_sample=False       # Greedy decoding
    )
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    if "### Response: [|AI|]" in response:
        return response.split("### Response: [|AI|]")[1].strip()
    return response

# ==== Query Documents ====
query = "كيف يتم توزيع الميراث إذا ترك المتوفى ابنة واحدة؟"
response = query_engine.query(query)

# Reformat response to build context for the prompt
context = "Context:\n"
for i in range(top_k):
    context += response.source_nodes[i].text + "\n\n"

print(context)

# ==== Construct and Process Prompt ====
full_prompt = prompt_ar.format_map({
    'context': context,
    'Question': query
})

final_answer = get_response(full_prompt)
print("== Final Answer ==\n", final_answer)