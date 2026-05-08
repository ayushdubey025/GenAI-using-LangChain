from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

text = "Bangalore is the capital of Karnataka"

vector = embedding.embed_query(text)

print(str(vector))

