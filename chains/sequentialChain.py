from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    task ="text-generation"
)

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic} \n',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a summary in 5 points, from the following text \n {text}',
    input_variables=['text']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1|model|parser |prompt2 |model|parser

result = chain.invoke({'topic':'Cinematography of Satyajit Ray'})
print(result)
chain.get_graph().print_ascii()