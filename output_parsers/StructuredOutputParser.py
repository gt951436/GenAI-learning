from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",  # by default,this model do not give structured output
    task ="text-generation"
)

model = ChatHuggingFace(llm=llm)

# schema to guide llm output
schema = [
    ResponseSchema(
        name='fact_1',
        description="Fact 1 about the topic"
    ),
    ResponseSchema(
        name='fact_2',
        description="Fact 2 about the topic"
    ),
    ResponseSchema(
        name='fact_3',
        description="Fact 3 about the topic"
    )
]

# parser creation
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'Fight Club'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)
print(type(final_result))

