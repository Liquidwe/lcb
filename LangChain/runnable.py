import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0.3,
)

outlinePromptTemplate = '''类型：{something_type}
列举 5 个该类型的东西'''

outlinePrompt = ChatPromptTemplate.from_template(outlinePromptTemplate)

tipsPromptTemplate = '''类型：{something_type}
根据提供的物品东西给出一个简短的描述。
'''

query = "水果"

tipsPrompt = ChatPromptTemplate.from_template(tipsPromptTemplate)
strParser = StrOutputParser()
outlineChain = outlinePrompt | model | strParser
tipsChain = tipsPrompt | model | strParser

# outline = outlineChain.invoke({"something_type": query})
# print(outline)
# tips = tipsChain.invoke({"theme": query})

map_chain = RunnableParallel(outline=outlineChain, tips=tipsChain)
print(map_chain.invoke({"something_type": query}))