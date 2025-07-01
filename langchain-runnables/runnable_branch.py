from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

# Prompt templates
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# LLM and parser
model = ChatOpenAI()
parser = StrOutputParser()

# Step 1: Generate report
report_gen_chain = RunnableSequence(prompt1, model, parser)

# Step 2: Count and print word count
word_count_step = RunnableLambda(lambda x: print(f" Word count: {len(x.split())}") or x)

# Step 3: If >500 words, summarize
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(
        RunnableLambda(lambda x: {"text": x}),  # Wrap into expected input
        prompt2, model, parser
    )),
    RunnablePassthrough()
)

# Final pipeline
final_chain = RunnableSequence(
    report_gen_chain,
    word_count_step,  # New word count step
    branch_chain
)

# Run
print(final_chain.invoke({'topic': 'Russia vs Ukraine'}))

# Optional: show graph
#final_chain.get_graph().print_ascii()
