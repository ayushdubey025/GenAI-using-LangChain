from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
# from numpy import np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills. His jersy number is 7 and it means thala for reason",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "who is Rohit Sharma??"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

results = cosine_similarity([query_embedding],doc_embeddings)[0]
index, score = sorted(list(enumerate(results)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print(f"Similarity score is: {score}")

