from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",  # by default,this model do not give structured output
    task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

# pydantic object --> schema ka kaam karega
class Person(BaseModel):
    name:str = Field(description='Name of the person')
    age:int = Field(gt=18,description='Age of the person')
    city:str = Field(description='Name of the city the person belongs to')
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the less-common name, age, city of a person belonging to {country}. \n {format_instruction}',
    input_variables = ['country'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


#prompt = template.invoke({'country':'India'})  # .format() also used
#print(prompt)

"""
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
"""

# using chains

chain = template|model|parser
result = chain.invoke({'country':'America'})
print(result)