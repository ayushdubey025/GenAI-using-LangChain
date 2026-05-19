
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# =========================================
# Create Documents
# =========================================

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# =========================================
# Initialize Chroma Vector Store
# =========================================

vector_store = Chroma(
    collection_name="sample",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="my_chroma_db"
)

# =========================================
# Add Documents
# =========================================

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

vector_store.add_documents(
    documents=docs,
    ids=ids
)

print("\nDocuments Added Successfully\n")

# =========================================
# View Stored Documents
# =========================================

stored_docs = vector_store.get(
    include=["documents", "metadatas"]
)

print("Stored Documents:\n")
print(stored_docs)

# =========================================
# Similarity Search
# =========================================

results = vector_store.similarity_search(
    query="Who among these are bowlers?",
    k=2
)

print("\nSimilarity Search Results:\n")

for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print(result.metadata)
    print("-----------------------------------")

# =========================================
# Similarity Search With Score
# =========================================

results_with_score = vector_store.similarity_search_with_score(
    query="Who are CSK players?",
    k=2
)

print("\nSimilarity Search With Score:\n")

for result, score in results_with_score:
    print(result.page_content)
    print("Score:", score)
    print("-----------------------------------")

# =========================================
# Metadata Filtering
# =========================================

filtered_results = vector_store.similarity_search(
    query="IPL players",
    filter={"team": "Chennai Super Kings"}
)

print("\nMetadata Filter Results:\n")

for result in filtered_results:
    print(result.page_content)
    print(result.metadata)
    print("-----------------------------------")

# =========================================
# Update Document
# =========================================

updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history and is regarded as one of the greatest T20 batsmen.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id="doc1",
    document=updated_doc1
)

print("\nDocument Updated Successfully\n")

# =========================================
# Delete Document
# =========================================

vector_store.delete(ids=["doc5"])

print("Document Deleted Successfully\n")

# =========================================
# Final Database State
# =========================================

final_docs = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nFinal Vector Store Data:\n")
print(final_docs)
