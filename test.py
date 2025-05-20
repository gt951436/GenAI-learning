from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model= 'gemini-2.0-flash',
                             model_kwargs = {"maxOutputTokens":512}
                            )

result = llm.invoke("What is the national bird of Equador?")
print(result.content)
