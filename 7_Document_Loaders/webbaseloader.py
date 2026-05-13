from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

from dotenv import load_dotenv 

load_dotenv()

url = "https://www.apple.com/in/iphone-17/"

loader = WebBaseLoader(url)

docs = loader.load()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "Give me answer of the following question : \n {question} \n based on the following context : \n {context}",
    input_variables = ['question', 'context']
)

# print(docs)

chain = prompt | model | parser

results = chain.invoke({'question': "what are the features and price of iphone 17?", "context": docs[0].page_content})

print(results)
