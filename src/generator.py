from transformers import pipeline

# Prompt template
TEMPLATE = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.

If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

# Call LLM with template
def generate_answer(question, chunks):
    context = "\n---\n".join([chunk[0]['product'] + ": " + chunk[0].get('chunk', 'N/A') for chunk in chunks])
    prompt = TEMPLATE.format(context=context, question=question)

    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=300)
    result = generator(prompt, do_sample=True, temperature=0.7)[0]['generated_text']

    return result
