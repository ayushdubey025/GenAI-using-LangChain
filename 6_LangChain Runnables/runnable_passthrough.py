from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# passthrough = RunnablePassthrough()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "tell me a joke about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "explain the joke - {text}",
    input_variables= ['text']
)

joke_chain = RunnableSequence(prompt1 | model | parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation' : RunnableSequence(prompt2 | model | parser)
    }

)

connector_Chain = RunnableSequence(joke_chain | parallel_chain)

results = connector_Chain.invoke({'topic': "cats"})

print(results['joke'])
print("-------------")
print(results['explanation'])