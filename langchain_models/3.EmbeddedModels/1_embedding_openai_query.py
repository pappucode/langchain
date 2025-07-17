from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32) # it needs low cost but low dimension contain less context

result = embedding.embed_query("Dhaka is the capital of India")

print(str(result))