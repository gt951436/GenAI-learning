from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path = "books",
    glob  = "*.pdf",
    loader_cls = PyPDFLoader
)
# loading 
#docs = loader.load()

#lazy loading
docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)