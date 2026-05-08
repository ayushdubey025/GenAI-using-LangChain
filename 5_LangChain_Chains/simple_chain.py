from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="generate 5 interesting lines about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

results = chain.invoke({"topic": 'dog'})

print(results)

chain.get_graph().print_ascii()