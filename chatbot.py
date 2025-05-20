from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
#from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="microsoft/Phi-3-mini-4k-instruct",task="text-generation")
model = ChatHuggingFace(llm=llm)

while True: # program runs until user exits
    user_input = input('You: ')
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI: ",result.content)
    