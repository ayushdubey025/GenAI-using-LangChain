from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

loader = DirectoryLoader(
    path = "books/",
    glob = "*.pdf",
    loader_cls=PyPDFLoader)

docs = loader.lazy_load()

# print(len(docs))
for doc in docs:
    print(doc.metadata)  


