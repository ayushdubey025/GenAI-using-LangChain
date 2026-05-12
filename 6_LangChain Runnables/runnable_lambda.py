from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser  = StrOutputParser()

# python function for calculate number of lines in a text.

def count_lines(text):
    count = len(text.split('\n'))  # splitting the text by newline character
    return count

prompt1 = PromptTemplate(
    template = "tell me a joke about {topic}",
    input_variables = ['topic']
)

joke_chain = RunnableSequence(prompt1 | model | parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'lines_of_joke': RunnableLambda(count_lines)
    }
)

connector_chain = RunnableSequence(joke_chain | parallel_chain)

results = connector_chain.invoke({'topic': "cow"})

final_results = """ {} \n word count : {} """.format(results['joke'], results['lines_of_joke'])

print(final_results)
