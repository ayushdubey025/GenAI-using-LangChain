from langchain_community.document_loaders import TextLoader, PyPDFLoader
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
    input_variables = ['text']
)

loader = PyPDFLoader("Ayush_Dubey_Resume.pdf")

docs = loader.load()

print(docs[0].meta_data)
print("-----------------------------------------------------------------")
print(len(docs))

chain = prompt | model | parser

results = chain.invoke({'text': docs[0].page_content})
print(results)