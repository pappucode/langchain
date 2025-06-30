
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['Positive', 'Negative'] = Field(description='Give the sentiment of the feedback.')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction' : parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feebback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feebback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could Not find sentiment !!")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a wonderfull phone.'})
print(result)

chain.get_graph().print_ascii()