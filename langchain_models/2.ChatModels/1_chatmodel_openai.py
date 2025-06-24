from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=1.5, max_completion_tokens=15)

result = model.invoke("give me 5 lines of poem about cricket.")

print(result.content)
