from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate(
    [('system', "You are expert in {domain}"),
    ('human', "Explain {topic} in simple terms")]
  
)

prompts = chat_template.invoke({'domain': "Data science", 'topic': "rag"})

print(prompts)