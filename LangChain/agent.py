from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from langchain_core.tools import BaseTool
from typing import Optional, Type, Dict, Any, List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel
import requests
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForToolRun

model_openapi = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-1106",
    temperature=0,
)


model_perplex = ChatPerplexity(temperature=0, verbose=True, model="llama-3-sonar-small-32k-online")


def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )


class Fetch_Top_Defi_Descriptions_Input(BaseModel):
    """Input for Fetch_Top_Defi_Descriptions_Input."""

    symbol: str = Field(description="Get the hottest defi protocol data.")


class Fetch_Top_GameFi_Descriptions_Input(BaseModel):
    """Input for Fetch_Top_GameFi_Descriptions_Input."""

    symbol: str = Field(description="Get the hottest gamefi protocol data.")


class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )


class FetchTopGameFiDescriptions(BaseTool):
    name: str = "fetch_top_gamefi_descriptions"
    args_schema: Type[BaseModel] = Fetch_Top_GameFi_Descriptions_Input
    description: str = """
  Get the hottest gamefi protocol data.

    Args:
      symbol (str): The symbol of the cryptocurrency to fetch.

  Returns:
      dict or str: If successful, returns a dictionary containing the data formatted for a line chart.
                   The data in this one is the most popular defi ranking.
                   If the data was successfully fetched, you want to expand the detail_description field of the data to at least 200 characters according to project.
                   If an error occurs, returns an error message as a string.

     Raises:
      Exception: If an unexpected error occurs during the API request.                
  """

    metadata: Optional[Dict[str, Any]] = {
        "description": "Query the hottest gamefi protocol data."
    }

    def _run(
            self,
            symbol: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        url = "https://api.chainbase.com/api/v1/custom/bcb15f821479b4d5/cryptocurrencies"

        payload = json.dumps({
            "sql": "SELECT a.*, b.short_description,b.detailed_description,b.ecosystem,b.tags,b.teams,b.fundraising,b.twitter_link,b.total_raised,b.launched_at FROM top_defi AS a LEFT JOIN rootdata AS b ON a.project ILIKE b.project_name limit 1",
            "page": 1,
            "pageSize": 10
        })
        headers = {
            'X-API-KEY': os.getenv("CHAINBASE_API_KEY"),
            'Content-Type': 'application/json'
        }
        print(f"Requesting {url} with data: {payload}")
        r = requests.request("POST", url, headers=headers, data=payload)
        print(r.json())
        return r.json()


class FetchTopDefiDescriptions(BaseTool):
    name: str = "fetch_top_defi_descriptions"
    args_schema: Type[BaseModel] = Fetch_Top_Defi_Descriptions_Input
    description: str = """
  Get the hottest defi protocol data.

    Args:
      symbol (str): The symbol of the cryptocurrency to fetch.

  Returns:
      dict or str: If successful, returns a dictionary containing the data formatted for a line chart.
                   The data in this one is the most popular defi ranking.
                   If the data was successfully fetched, you want to expand the detail_description field of the data to at least 200 characters according to project.
                   If an error occurs, returns an error message as a string.

     Raises:
      Exception: If an unexpected error occurs during the API request.                
  """

    metadata: Optional[Dict[str, Any]] = {
        "description": "Query the hottest defi protocol data."
    }

    def _run(
            self,
            symbol: str,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        url = "https://api.chainbase.com/api/v1/custom/bcb15f821479b4d5/cryptocurrencies"

        payload = json.dumps({
            "sql": "SELECT a.*, b.short_description,b.detailed_description,b.ecosystem,b.tags,b.teams,b.fundraising,b.twitter_link,b.total_raised,b.launched_at FROM top_defi AS a LEFT JOIN rootdata AS b ON a.project ILIKE b.project_name",
            "page": 1,
            "pageSize": 10
        })
        headers = {
            'X-API-KEY': os.getenv("CHAINBASE_API_KEY"),
            'Content-Type': 'application/json'
        }
        print(f"Requesting {url} with data: {payload}")
        r = requests.request("POST", url, headers=headers, data=payload)
        print(r.json())
        return r.json()


def pp_online():

    promptTemplate02 ='''
    {input}
    
    需要将格式化的回答转换成一个 markdown 表格，包含以下字段：
    - 项目名称
    - 本周活跃用户数
    - 上周活跃用户数
    
    在表格后面加上每个项目的详细介绍，详细介绍下什么是这个项目，它的目标是什么，它的优势是什么，它的团队是谁，它的融资情况如何，它的 Twitter 链接是什么，它的总筹资是多少，它是什么时候发布等等。
    
    比如:
| project    | percentage_change | current_week_total_amount | current_week_unique_contracts |
|------------|-------------------|---------------------------|-------------------------------|
| uniswap   | 50                | 1381080609.17538          | 186                           |
| pancake     | 40                | 28244054.918360148        | 474                          |
| apeswap    | 37                | 16825791.636527166        | 91                          |
| babyswap   | 23                | 47590652.06893177         | 41                           |
| velodrome  | 12                | 57745073.39169332         | 31                           |

    项目介绍
    第一名：uniswap
    Uniswap 是一个去中心化的交易所（DEX），基于以太坊区块链，允许用户无需中介机构即可交换加密货币。其独特之处在于使用了自动化做市商（AMM）模型，通过流动性池来定价和交易资产。

以下是 Uniswap 的主要特点：

- **自动化做市商（AMM）模型**：Uniswap 采用 AMM 模型，使用公式（通常是 x * y = k）来确定代币的价格。这意味着价格由池中的代币数量决定，而不是通过订单簿。
- **流动性池**：用户可以将两种代币存入流动性池中，成为流动性提供者（LP），并根据交易产生的费用赚取一定比例的收益。
- **无需许可**：任何人都可以在 Uniswap 上创建新的流动性池或为现有池提供流动性，无需许可。
- **去中心化**：Uniswap 是完全去中心化的，没有中心化的管理机构。智能合约自动执行所有操作，交易在链上公开透明。
- **Uniswap 代币（UNI）**：Uniswap 有自己的治理代币 UNI，持有者可以参与 Uniswap 的治理决策，例如投票决定协议的改进或调整交易费率。

3. **项目团队：**
    
    **Founder and CEO:** Hayden Adams 
    
    **COO:** Mary Catherine (MC) Lader 
    
    **Chief Legal Officer:** Marvin Ammori 
    
    ……（信息全部列完）
    
4. **项目融资：**
    
    '''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名专业的区块链助手。你需要判断是否需要使用工具来完成用户的问题，如果不需要使用工具可以直接回答。"),
            ("user", promptTemplate02),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    tools = [FetchTopDefiDescriptions(), FetchTopGameFiDescriptions()]
    agent = create_openai_tools_agent(model_openapi, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


if __name__ == '__main__':
    model = pp_online()
    print(model.invoke({"input": "最火的gamefi?"}))
    # print(model.invoke({"input": "Consensys 有哪些产品"}))
    # print(model.invoke({"input": "zksync的空投表现如何?"}))
    # print(model.invoke({"input": "什么是Thruster?"}))