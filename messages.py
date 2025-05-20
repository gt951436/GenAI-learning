from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id = "microsoft/Phi-3-mini-4k-instruct",task = "text-generation")
model = ChatHuggingFace(llm = llm)

# human message --> user to llm
# AI message --> llm to user response
# system message --> initial instructions to AI

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="Tell me about LangChain")
]
res = model.invoke(messages)
messages.append(AIMessage(content=res.content))

print(messages)