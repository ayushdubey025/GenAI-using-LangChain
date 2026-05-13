from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv 

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(
    template = "summarize the following text : \n {text}",
    input_variables= ['text']
)

loader = TextLoader("cricket.txt", encoding = "utf-8")

docs = loader.load()

print(docs)

print(len(docs[0].page_content))

print(docs[0])

chain = prompt | model | parser

results = chain.invoke({'text': docs[0].page_content})

print(results)