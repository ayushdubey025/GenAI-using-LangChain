from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

docs = [
    "Bangalore is capital of Karnataka",
    "Lucknow is the capital of U.P.",
    "Chandigarh is the capital of Haryana"
]

result = embeddings.embed_documents(docs)

print(str(result))