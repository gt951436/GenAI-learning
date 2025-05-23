from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",  # by default,this model do not give structured output
    task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

#parser creation
parser = JsonOutputParser()

# prompt template
template = PromptTemplate(
    template = 'Give me the name,age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    
    # partial variable --> runtime se pehle hi fill hojaata hai
    partial_variables={'format_instruction':parser.get_format_instructions()} # Return a JSON object
)
prompt = template.format()
#print(prompt)

result = model.invoke(prompt)
#print(result)

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))  #dict type
print(final_result['name'])