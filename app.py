import streamlit as st
from rag_pipeline import retrieve_relevant_chunks, generate_answer

st.set_page_config(page_title="Complaint AI Assistant", layout="wide")

st.title("📨 Complaint AI Assistant")
st.write("Ask any question based on customer complaints.")

# Text input
user_question = st.text_input("Your Question:", "")

# Submit button
if st.button("Ask") and user_question.strip():
    with st.spinner("Thinking..."):
        chunks = retrieve_relevant_chunks(user_question)
        answer, sources = generate_answer(user_question, chunks)
        
        # Display Answer
        st.markdown("### 🤖 AI Answer")
        st.write(answer)
        
        # Display Sources
        st.markdown("### 🔎 Retrieved Complaint Excerpts")
        for i, source in enumerate(sources):
            st.markdown(f"**Source {i+1}:**")
            st.write(source.page_content)
            st.caption(f"Product: {source.metadata.get('product')}, Complaint ID: {source.metadata.get('complaint_id')}")

# Clear Button
if st.button("Clear"):
    st.experimental_rerun()
