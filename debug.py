# debug.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/Solar_energy_0105.txt", encoding="utf-8")
docs = loader.load()
print("Full content length:", len(docs[0].page_content))
print("First 300 chars:")
print(docs[0].page_content[:300])
print("\n--- Chunks ---")
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks[:3]):
    print(f"\nChunk {i+1} ({len(c.page_content)} chars):")
    print(c.page_content[:200])
