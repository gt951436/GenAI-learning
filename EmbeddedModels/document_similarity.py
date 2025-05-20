import pprint
from langchain_huggingface import HuggingFaceEmbeddings  # new, supported path :contentReference[oaicite:1]{index=1}
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# Initialize with a Sentence-Transformers model from HF Hub
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy canine.",
    "Artificial intelligence and machine learning are transforming industry.",
]
query = "tell me about fox"

# Embed documents and query
doc_embeds   = embedding.embed_documents(documents)
query_embed  = embedding.embed_query(query)

pp = pprint.PrettyPrinter(indent=2, width=120)
print("\nDocument Embeddings:")
for idx, vec in enumerate(doc_embeds):
    print(f"\nDocument {idx} embedding (length {len(vec)}):")
    pp.pprint(vec)

print("\nQuery Embedding:")
print(f"(length {len(query_embed)})")
pp.pprint(query_embed)

# Compute similarity
print("\nCosine Similarities:")
scores = cosine_similarity([query_embed], doc_embeds)[0]
for idx, score in enumerate(scores):
    print(f"Similarity(query, doc{idx}): {score:.4f}")
