from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="microsoft/Phi-3-mini-4k-instruct",task="text-generation")
model = ChatHuggingFace(llm=llm)

# Static messages in case of list of msges
# chat history
chat_history = [
    SystemMessage(content = "You are a helpful AI assistant")
]

while True: # program runs until user exits
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ",result.content)
    
print("chat History\n", chat_history)
    