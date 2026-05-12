from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser  = StrOutputParser()

# python function for calculate number of words in a text.

def count_words(text):
    count = len(text.split(' '))  # splitting the text by whitespace
    return count

prompt1 = PromptTemplate(
    template = "tell me about {topic}",
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = "explain the topic in 5 points. \n {text}",
    input_variables = ['text']
)

topic_chain = RunnableSequence(prompt1 | model | parser)

branch_chain = RunnableBranch(
    (lambda x: count_words(x) >= 500, RunnableSequence(prompt2 | model | parser)),
    RunnablePassthrough()
)

connector_chain = RunnableSequence(topic_chain | branch_chain)

results = connector_chain.invoke({'topic': "movies"})

print(results)