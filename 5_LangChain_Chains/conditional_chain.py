from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the sentiments of feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template = "classify the sentiments of feedback and tell positive and negative \n {feedback} \n {format_instructions}",
    input_variables = ['feedback'],
    partial_variables = {'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser

# positive feedback
prompt2 = PromptTemplate(
    template = "write an appropriate response on positive feedback \n {feedback}",
    input_variables = ['feedback']
)

# negative feedback
prompt3 = PromptTemplate(
    template = "write an appropriate response on negative feedback \n {feedback}",
    input_variables = ['feedback']
)

# conditional chaining
branch_chain = RunnableBranch(
    (lambda x:x.get("sentiment") == 'positive', prompt2 | model | parser),
    (lambda x:x.get("sentiment")  == 'negative', prompt3 | model | parser),
    RunnableLambda (lambda x: "cound find any sentiments")
)



chain = classifier_chain | branch_chain

results = classifier_chain.invoke({"feedback":"This product is damaged when recieved."})

print(results)

chain.get_graph().print_ascii()