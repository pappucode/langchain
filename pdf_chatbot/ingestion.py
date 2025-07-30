# Data Ingestion
# Step 1 — Importing Libraries

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Step 2 — Loading the PDF Content
loader = PyPDFLoader("impact_of_generativeAI.pdf")
document = loader.load()

# print(len(document))
# print(document)
#print(docs[0])
#print(document[0].page_content)
#print(document[0].metadata)


# Step 3 — Splitting Documents into Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(document)
#print(f"created {len(texts)} chunks")
#print(texts[0])
#print(texts[0].page_content)

#Step 4 — Creating Embeddings and Storing in Pinecone
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))