from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("7_Document_Loaders/Ayush_Dubey_Resume.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
    # separators = ['\n\n', '\n', ' ','']
)

split_docs = splitter.split_documents(docs)

print(len(split_docs))

print(split_docs)