# rag_pipeline.py

def retrieve_relevant_chunks(question, retriever, k=5):
    """
    Embeds the user question and retrieves top-k most relevant text chunks from the vector store.
    """
    docs = retriever.similarity_search(question, k=k)
    return docs

def generate_answer(question, context_chunks, llm):
    """
    Constructs a prompt with the retrieved chunks and sends it to the LLM to generate an answer.
    """
    context = "\n\n".join([doc.page_content for doc in context_chunks])
    
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
    """.strip()

    response = llm.invoke(prompt)
    return response, context_chunks
