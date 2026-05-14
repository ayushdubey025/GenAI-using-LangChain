from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

loader = PyPDFLoader("7_Document_Loaders/Ayush_Dubey_Resume.pdf")

docs = loader.load()

split_docs = splitter.split_documents(docs)

print(split_docs[3])