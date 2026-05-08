from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI()

chatHistory = [
    SystemMessage(content="You are the best teacher")
]


while True:
    user_input = input("You: ").strip()
    chatHistory.append(HumanMessage(content=user_input))


    if user_input.lower() == "exit":
        break

    # currentChat = {}
    # currentChat["You"] = user_input
    

    # results = model.invoke(currentChat["You"])
    # currentChat["AI"] = results.content

    # chatHistory.append(currentChat)



    results = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=results.content))
    print(f"AI: {results.content}")

print(chatHistory)