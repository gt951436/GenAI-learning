from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """There was only one way to do things in the Statton house. 
That one way was to do exactly what the father, Charlie, demanded. 
He made the decisions and everyone else followed without question. 
That was until today.

It was just a burger. Why couldn't she understand that? 
She knew he'd completely changed his life around her eating habits, so why couldn't she give him a break this one time? 
She wasn't even supposed to have found out. Yes, he had promised her and yes, 
he had broken that promise, but still in his mind, all it had been was just a burger.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)