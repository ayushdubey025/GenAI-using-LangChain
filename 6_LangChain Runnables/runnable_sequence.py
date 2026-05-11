from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "write any {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "who said {text}",
    input_variables= ['text']
)


chain = RunnableSequence(prompt1 | model | parser | prompt2 | model | parser)

results = chain.invoke({"topic":"quote"})

print(results)