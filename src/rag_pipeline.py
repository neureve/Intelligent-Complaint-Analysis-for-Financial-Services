from retriever import retrieve_top_k
from generator import generate_answer

def answer_question_with_rag(question, k=5):
    chunks = retrieve_top_k(question, k=k)
    for c in chunks:
        c[0]['chunk'] = c[0].get('chunk', 'N/A')  # Ensure chunk content for prompt
    answer = generate_answer(question, chunks)
    return answer, chunks
