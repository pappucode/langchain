import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Step 1 — Loading Your Knowledge
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Step 2 — Building Your RAG Chains and Asking Questions
llm = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
) 

res = qa.invoke("What are the applications of generative AI according the the paper? Please number each application.")
print(res) 

res = qa.invoke("Can you please elaborate more on application number 2?")
print(res)