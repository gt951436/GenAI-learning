from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
#from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="microsoft/Phi-3-mini-4k-instruct",task="text-generation")
model = ChatHuggingFace(llm=llm)

# chat history
chat_history = []

while True: # program runs until user exits
    user_input = input('You: ')
    chat_history.append(user_input)
    
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ",result.content)
    
print("chat History\n", chat_history)
    