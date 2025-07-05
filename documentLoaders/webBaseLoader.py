from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = 'Answer the following questions {question} from the text : {text}',
    input_variables = ['question','text']
)
parser = StrOutputParser()

urls = ["https://www.geeksforgeeks.org/python/iterators-in-python/",
           "https://huggingface.co/models?pipeline_tag=text-generation&sort=trending&search=meta+"
        ]
loader = WebBaseLoader(urls)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'what is the use and benefits of iterators in python?','text': docs[0].page_content}))



