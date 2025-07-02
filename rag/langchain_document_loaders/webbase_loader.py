from langchain_community.document_loaders import WebBaseLoader
import re
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://shopping.indiamart.com/products/?id=2853361138412&pos=4&kwd=ladies%20western%20wear&tags=A'

loader = WebBaseLoader(url)

docs = loader.load()

#print(len(docs))
# raw_text = docs[0].page_content
# # Remove excessive whitespace
# cleaned_text = re.sub(r'\s+', ' ', raw_text)        # Replace multiple whitespace with single space
# cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text)  # Keep paragraph breaks
# cleaned_text = cleaned_text.strip()
#print(cleaned_text)

chain = prompt | model | parser

result = chain.invoke({'question': 'What is the product that we are talking about?', 'text': docs[0].page_content})

print(result)