from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

from dotenv import load_dotenv 

load_dotenv()

load = CSVLoader("Social_Network_Ads.csv")

docs = load.load()

print(docs)

model = ChatOpenAI()

parser = StrOutputParser() 

prompt = PromptTemplate(
    template = "Give me answer of the following question : \n {question} \n based on the following context : \n {context}",
    input_variables = ['question', 'context']
)

chain = prompt | model | parser

results = chain.invoke({"question": "what is minimum age for estimated salary is greater than 30000?", "context": docs})