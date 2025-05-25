from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",  # by default,this model do not give structured output
    task ="text-generation"
)

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic} \n',
    input_variables=['topic']
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt|model|parser

result = chain.invoke({'topic':'Cinematography of Satyajit Ray'})
print(result)
chain.get_graph().print_ascii()