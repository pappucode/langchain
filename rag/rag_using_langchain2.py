from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#print(vector_store.index_to_docstore_id)
#print(vector_store.get_by_ids(['1d53e418-65c5-4555-bca3-70f9695c3a5b']))

## Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k":4})
#print(retriever) 

#print(len(retriever.invoke("What is deepmind?")))
#print(retriever.invoke("What is deepmind?"))

## Step 3 - Augmentation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=['content', 'question']
)

question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)
#print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#print(context_text)

final_prompt = prompt.invoke({'context':context_text, 'question':question})
#print(final_prompt)

##Step 4 - Generation
answer = llm.invoke(final_prompt)
#print(answer.content)

## Building a Chain