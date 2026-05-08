from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messeges = [SystemMessage(content="You are best teacher"),
            HumanMessage(content="what is 2 plus 2?")]

result = model.invoke(messeges)

messeges.append(AIMessage(content=result.content))

print(messeges)

