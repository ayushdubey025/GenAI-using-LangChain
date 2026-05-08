from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()



prompt1 = PromptTemplate(
    template="how many letters in {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='multiply the {first_result} by 10',
    input_variables=['first_result']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

results = chain.invoke({"topic": 'Anaconda'})

print(results)

chain.get_graph().print_ascii()