from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):

    summary : str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.There are too many pre-installed apps that I can't remove. Also, the UI looks outdated campared to other brands. Hoping for a software update to fix this.""")

print(result)

