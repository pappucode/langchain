from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='EleutherAI/gpt-neo-125M',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100
    )
)

#model = ChatHuggingFace(llm=llm)
response = llm.invoke("What is the capital of Bangladesh?")
print(response)
