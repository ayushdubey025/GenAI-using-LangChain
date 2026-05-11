from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')


parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "generate a tweet about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = "generate a linkedin post about {topic}",
    input_variables= ['topic']
)

chain1 = RunnableParallel({
    'tweet': RunnableSequence(prompt1 | model1 | parser),
    'linkedin_post': RunnableSequence(prompt2 | model2 | parser)
})

results = chain1.invoke({'topic': "AI"})

print(results['tweet'])
print("-------------")
print(results['linkedin_post'])
chain1.get_graph().print_ascii()