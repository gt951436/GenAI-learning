from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./Nvidia_LLVM_intern.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator=""
)
chunks = splitter.split_documents(docs)
print(chunks[1])