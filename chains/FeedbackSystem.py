from langchain_huggingface import ChatHuggingFace,  HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.0-flash',
    temperature=0,
    system_message=(
        "You are a customer-service assistant. "
        "Whenever you get a piece of feedback, "
        "just generate the reply—do NOT ask for more details or clarification."
    ),
)

"""AI feedback response system based on the sentiment of feedbacks provided by customers"""

parser1 = StrOutputParser()

"""LLM's output is not controlled so we need to somehow make the output of 
the classifier-chain consistent becoz next chain inputs depend on it's output.
"""
# output ko structured banaana hoga --> pydantic output schema
#structured output schema
class Feedback(BaseModel):
    sentiment : Literal['Positive','Negative'] = Field(description='Give the sentiment of the feedback')
    
parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1|model|parser2
classifier_chain.get_graph().print_ascii()

"""result = classifier_chain.invoke({'feedback':'this software is Bogus'})
print(result)"""

# positive feedback --> prompt for further process
prompt2 = PromptTemplate(
    template = ("You are a customer-service agent.  "
        "Here's some positive feedback; write a friendly thank-you reply in 2–3 sentences.  "
        "Do NOT ask for any more information.\n {feedback}"),
    input_variables=['feedback']
)

# negative feedback --> prompt for further process
prompt3 = PromptTemplate(
    template = ("You are a customer-service agent."
        "Here's some negative feedback; write an empathetic apology and offer a solution"
        "in 2-3 sentences. Do NOT ask for any more information.\n \n {feedback}"),
    input_variables=['feedback']
)

#Branch chain on the parsed sentiment
branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='Positive',prompt2|model|parser1),
    (lambda x:x.sentiment=='Negative',prompt3|model|parser1),
    RunnableLambda(lambda x: "couldn't find any sentiment!")  # default case
)
branch_chain.get_graph().print_ascii()

final_chain = classifier_chain|branch_chain

answer = final_chain.invoke({'feedback':'The recent update in software is amazing but the hardware is older and not good enough.'})
print(answer)
final_chain.get_graph().print_ascii()
