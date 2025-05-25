from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    task ="text-generation"
)
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id = "sarvamai/sarvam-m",
    task ="text-generation"
)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 5 short questions answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes --> {notes} and quiz --> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

# parallel chaining --> RunnableParallel used
parallel_chain = RunnableParallel(
    {
        'notes': prompt1|model1|parser,
        'quiz': prompt2|model2|parser
    }
)
merge_chain = prompt3|model1|parser

final_chain = parallel_chain|merge_chain

text = """
Querying a machine learning model on its own, without a vector database, is neither fast nor cost-effective. Machine learning models cannot remember anything beyond what they were trained on. They have to be the context every single time (which is how many simple chatbots work).
Passing the context of a query to the model every time is very slow, as it is likely to be a lot of data; and expensive, as data has to move around, and computing power has to be expended repeatedly having the model parse the same data. And in practice, most machine learning APIs are likely constrained in how much data they can accept at once anyway.
This is where a vector database comes in handy: a dataset goes through the model only once (or periodically as it changes), and the model's embeddings of that data are stored in a vector database.
This saves a tremendous amount of processing time. It makes building user-facing applications around semantic search, classification, and anomaly detection possible, because results come back within tens of milliseconds, without waiting for the model to crunch through the whole data set.
For queries, developers ask the machine learning model for a representation (embedding) of just that query. Then the embedding can be passed to the vector database, and it can return similar embeddings — which have already been run through the model. Those embeddings can then be mapped back to their original content: whether that is a URL for a page, a link to an image, or product SKUs.
To summarize: Vector databases work at scale, work quickly, and are more cost-effective than querying machine learning models without them.
"""
result = final_chain.invoke({'text':text})
print(result)