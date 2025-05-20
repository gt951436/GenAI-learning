from langchain_core.prompts import ChatPromptTemplate

# Dynamic messaging in case of multi-turn messaging

chat_template = ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}?'),  
])

prompt = chat_template.invoke({'domain':'Cricket','topic':'Doosra'})
print(prompt)