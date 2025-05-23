from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",  # by default,this model do not give structured output
    task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1-st prompt --> detailed report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

# 2-nd prompt --> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1.content})
result2 = model.invoke(prompt2)

print(result2.content)