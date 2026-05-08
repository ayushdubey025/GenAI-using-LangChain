from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

llm = HuggingFaceEndpoint(
repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model2 = ChatHuggingFace(llm=llm)

model3 = ChatAnthropic(model='claude-3-5-sonnet-20241022')


prompt1 = PromptTemplate(
    template="make notes on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='generate 5 questions on {topic}\n',
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template='merge both notes and questions in single document. \n Notes -> {notes} \n Quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser

})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

results = chain.invoke({'topic': "galaxy"})
print(results)

chain.get_graph().print_ascii()