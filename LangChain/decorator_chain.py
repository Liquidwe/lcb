from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-1106",
    temperature=0,
)
model_perplex = ChatPerplexity(temperature=0, verbose=True, model="llama-3-sonar-small-32k-online")

prompt1 = ChatPromptTemplate.from_template("随机给出一种 {topic}")
prompt2 = ChatPromptTemplate.from_template("{story}\n\n写一段简短的描述")

prompt = PromptTemplate.from_template(
        """鉴于下面的用户问题，将其分类为“LangChain”、“OpenAI”或“其他”。

不要用超过一个字来回应。

<question>
{question}
</question>

分类："""
)

chainA = (
    prompt
    | model
    | StrOutputParser()
)

langchainPrompt = PromptTemplate.from_template(
        """您是 langchain 方面的专家。 
回答问题时始终以“正如老陈告诉我的那样”开头。 
回答以下问题：

问题：{question}
回答："""
)
langchain_chain = langchainPrompt | model

OpenAIPrompt = PromptTemplate.from_template(
        """您是 OpenAI 方面的专家。 
回答问题时始终以“正如奥特曼告诉我的那样”开头。 
回答以下问题：

问题：{question}
回答："""
)
OpenAI_chain = OpenAIPrompt | model

generalPrompt = PromptTemplate.from_template(
        """ 回答以下问题：

问题：{question}
回答："""
)
general_chain = generalPrompt | model


def route(info):
    if "OpenAI" in info["topic"]:
        return OpenAI_chain
    elif "LangChain" in info["topic"]:
        return langchain_chain
    else:
        return general_chain


@chain
def custom_chain3(text):
    c = prompt1 | model | StrOutputParser() | {"story": RunnablePassthrough()} | prompt2 | model_perplex | StrOutputParser()
    return c.invoke({"topic": text})


if __name__ == '__main__':
    # print(custom_chain3.invoke("水果"))
    full_chain = {"topic": chainA, "question": lambda x: x["question"]} | RunnableLambda(route)
    print(full_chain.invoke({"question": "我如何使用OpenAI的模型?"}))
