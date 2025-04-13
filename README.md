# Disclaimer 
This project is for educational and research purposes only. All responses must be verified by qualified scholars before real-world application.

# Shariah complaint AI chat bot

This project is an (arabic) AI chat bot that specializes in responding to queries relating to inheritance and other money related questions. The chat bot uses shariah law by taking in 
context from the quran and hadith as well as data from islamic inheritance-related books. 

It retrieves the most relevant sources from a structured JSON database and uses DeepSeek’s Arabic language model to generate scholarly, source-backed answers in Arabic.

## Why this is important

Responses to islamic questions require verified and reliable evidence, which is in the form of quran and hadith. Before responding to questions, this chat bot makes sure to
read the relevant evidence from the database, and answer based on the evidence it collected and the context of the question. If no releveant evidence is found based on the
user's question, the chat bot does not give an answer, but gives a response stating that it cannot answer the question. When responding, the chat bot cites the evidence it 
used to answer the question, which can be fact checked by the user. 

## How it works
1. **Preprocessed Data**: A single, unified JSON file combines Quran, Hadith, and inheritance rules.
2. **Document Embedding**: Semantic embeddings are created using `intfloat/multilingual-e5-base` and stored in a FAISS index.
3. **Query Matching**: User questions are semantically matched with top relevant documents.
4. **RAG Prompting**: A context-aware prompt is constructed and sent to DeepSeek for response generation.
5. **Citations**: If the answer includes Quran or Hadith, the source is explicitly referenced.

The main implementation is in "deep.py". Simply install the requirements, adjust the query, and run the code!. 
