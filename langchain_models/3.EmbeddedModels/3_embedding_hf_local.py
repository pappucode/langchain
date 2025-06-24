from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Dhaka is the capital of Bangladesh.",
    "Chittagong is the port city of Bangladesh.",
    "Paris is the capital of France."
]  
result = embedding.embed_documents(documents)
print(str(result))