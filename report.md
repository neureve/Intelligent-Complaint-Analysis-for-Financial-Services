3. Report Section (for README or submission)
🔹 Text Chunking Strategy
To improve semantic retrieval, we implemented a chunking strategy using LangChain's RecursiveCharacterTextSplitter. We set chunk_size = 500 and chunk_overlap = 50 after experimenting with various settings.

Why chunking? Embedding full documents as single vectors often leads to poor search results, especially for long complaints.

Why these values? 500 tokens provides sufficient context, while 50 tokens overlap ensures context continuity across chunks.

Justification: This balance ensures semantic granularity while preserving readability for downstream retrieval.

🔹 Embedding Model Choice
We used sentence-transformers/all-MiniLM-L6-v2, a compact and fast transformer model known for:

High performance on semantic similarity tasks

Low memory requirements (suitable for local environments)

Good performance in practical QA and chunk-based search applications

This makes it an ideal choice for our vector-based complaint search system.