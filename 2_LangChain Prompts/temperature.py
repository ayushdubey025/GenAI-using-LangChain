from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=1.5)

query = "tell me 4 lines poem on cat"
result = model.invoke(query)

print(result.content)