
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from getCustomerDataset import  getCsvDataset

from getSummary import getSummary
from getFAISS import getFAISS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage

from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor

import os

# === LLM ===
from langchain_community.llms import Ollama

# llm = Ollama(model = "llama3", temperature = 0, num_predict=320)
# llm2 = Ollama(model = "gemma:2b", temperature = 0.2)
llm = Ollama(model = "gemma2", top_k=1) #llama3-chatqa:8b, qwen:4b(Alibaba), aya:8b(Cohere), phi3 (3.8B)(Microsoft), gemma:2b zhaoyun/phi3-128k-chinese
llm2 = Ollama(model = "gemma2", top_k=1)
llm3 = Ollama(model = "mistral", top_k=1)

SummaryFunc = getSummary(llm) # 宣告一個儲存資料和做總結的物件
RetrievalQA = getFAISS(llm)


# def get_fields(directory):    
#     with open(directory+"tables.txt", 'r') as file:
#         result = file.read()
    
#     return result

def get_fields(lines, table_name):    
    result = ''
    for field in lines[1:]:
        if ((table_name.strip()) in field.split('|')[0]):
            result += (field+'\n')

    return result




def get_metadata(directory):    
    with open(directory+"metadata.txt", 'r', encoding='utf-8') as file:
        result = file.read()
    
    return result

def get_tables_name(directory):
    name_list = []
    for csv in os.listdir(directory):
        csv_name, csv_extentsion = csv.split('.')
        if  csv_extentsion == "csv":
            name_list.append(csv_name)
    
    return ", ".join(name for name in name_list)


def parse_question(msg):
    system = f'''
        parse question into key points, then output answer as format in chinese.
        Ex:
        Q. 給我所有基金和債券的商品名稱
        1.商品名稱

        Q. 給我所有基金或債券形式的商品名稱
        1. 商品名稱
        2. 商品形式

        Q. 請給我每個員工在2023年內的平均業績
        1. 每個員工
        2. 2023年
        3. 平均業績

        Q. 在2023/02/24前，員工吳光磊跟客戶王曉明的交易金額>20000的資料
        1. 2023/02/24
        2. 員工吳光磊
        3. 客戶王曉明
        4. 交易金額>20000

        Q. 請給我2023年業績最高的員工
        1. 2023年
        2. 業績最高
        3. 員工
    '''

    human = '''{input}'''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    print("Processing ...")
    return llm.invoke(prompt.format(input=msg))

def generate_tables_name(table_inform, msg, llm):
    system = f'''
        {table_inform}

        Metadata shows what fields include in that table.
        List all Table Name to be queried and JOIN, not SQL.
        Ex: table1, table2, table3
        No more explanations or messages, only Table Name and split by ','.
    '''

    human = '''{input}'''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    print("Processing ...")
    return llm.invoke(prompt.format(input=msg))


def generate_SQL(table_inform, msg, llm):
    print("input: " + msg + "\n")
    system_sql = f'''

    {table_inform}

    First, Follow the example:
    1. 問題中有與'總數'、'總'相關字詞： SELECT SUM(table_name.field_name) FROM table GROUP BY table_name.field_name;
    2. 問題中有與'筆數'相關字詞: SELECT COUNT();
    3. 問題要求'包含 field1 field2' : SELECT table_name.field_name1, table_name.field_name2;
    4. 問題需要不只一個 table 的欄位： FROM table_name1 JOIN table_name1.field_name1 ON table_name2.field_name2;
    5. 問題中有與日期相關字詞： WHERE dates condition;
    6. 問題中有與'最高'、'最低'相關字詞： ORDER BY table_name.field_name (DESC or ASC) LIMIT n;

    Then, answer as format 
    ```sql
        SELECT (DISTINCT) table_name.field_name FROM table_name;
    ```, SQL command must end by ';'
    '''
    # 1. Give the Date field as format 'YYYYMMDD', ex. Date field = '20240101'     2. SELECT 'table_name.field_name' in SQL.
    # 2. If input question said before the Date, then let Date field <= '20240101' in SQL command
    # 4. Use SELECT * when user ask the entire data.
    # 2. Use SELECT * when user ask the entire data.
    # 1. May need to join the tables, when the field names in WHERE is not included in tables.
    # 2. Binder Error: Table 'table' does not have a column named 'XXX'
    # 3. Catalog Error: Table with name 'XXX' does not exist!
    # Must SELECT all field which in GROUP BY.
    # Thought First:
    # 依照以下格式，解析問題並填入答案
    # 1.欲使用的條件: XXX (Which in WHERE Conditions)
    # 2.欲使用的計算: XXX (Which is Function for computation)

    human_sql = '''{input}

        (Please respond a SQL Command)'''

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_sql),
            ("human", human_sql),
        ]
    )
    print("Processing ...")
    return llm.invoke(prompt.format(input=msg))

def query_by_SQL():
    tables = generate_tables_name(get_metadata(SummaryFunc.database), SummaryFunc.Input_Question, llm2) # Ollama(model = model2, top_k=1)
    # print(f'Tables: {tables}')
    
    with open(SummaryFunc.database+"tables.txt", 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')

    table_inform = f'table: {tables}\n{lines[0]}\n'

    for table_name in tables.split(','):
        # print(f"--table--: {table_name}\n")
        if table_name.strip() != '':
            table_inform += get_fields(lines, table_name) + "\n"

    print(table_inform)
    sqlCommand = generate_SQL(table_inform, SummaryFunc.Input_Question, llm3)  # Ollama(model = model3, top_k=1)
    sqlCommand = sqlCommand.replace('`', '')
    print(f"LLM Output: {sqlCommand}")
    return SummaryFunc.store_dataset(getCsvDataset.dataset_query_sql(SummaryFunc.database, sqlCommand))

def reload():
    # === searchOracleInput ===
    class SearchOracleInput(BaseModel):
        action_input: str = Field(description=f'''
            .
        ''') # Expected a Input Question as String.

    def search_from_oracle(action_input: str) -> str:

        SummaryFunc.database = './database/Oracle/'
        SummaryFunc.Action_Output = action_input
        SummaryFunc.for_sql = 'True'

        return parse_question(SummaryFunc.Input_Question) + "\n\n 請問以上是否正確? 正確，請回答'1'，不正確，請回答'2'。"

    searchFromOracle = StructuredTool.from_function(
        func=search_from_oracle,
        name="search from Oracle",
        description= f"{get_metadata('./database/Oracle/')}, if required data is relative, use 'search from Oracle'.",
        args_schema=SearchOracleInput,
        return_direct=True,
        # coroutine= ... <- you can specify an async method if desired as well
    )
    # === searchFileInput ===
    class SearchFileInput(BaseModel):
        action_input: str = Field(description=f'''
            Metadata shows what fields include in that table.
            List all Table Name to be queried and JOIN, not SQL.
            Ex: table1, table2, table3
            No more explanations or messages, only Table Name.
        ''')

    def search_from_file(action_input: str) -> str:

        SummaryFunc.for_sql = 'False'
        result = RetrievalQA.invoke(SummaryFunc.Input_Question)

        return result
        

    searchFromFile = StructuredTool.from_function(
        func=search_from_file,
        name="search from File",
        description=f"{get_metadata('./database/File/')}, if required data is relative, use 'search from File'.",
        args_schema=SearchFileInput,
        return_direct=True,
    )

    # === searchSQLInput ===
    class SearchSQLInput(BaseModel):
        action_input: str = Field(description=f'''
        .
        ''')
        # Parse out the values, strings, dates, etc. in the question,
        # And return corresponding answers in the following format as string directly:

    def search_from_sql(action_input: str) -> str:
        
        SummaryFunc.database = './database/SQL/'
        SummaryFunc.Action_Output = action_input
        SummaryFunc.for_sql = 'True'

        return parse_question(SummaryFunc.Input_Question) + "\n\n 請問以上是否正確? 正確，請回答'1'，不正確，請回答'2'。"
        

    searchFromSQL = StructuredTool.from_function(
        func=search_from_sql,
        name="search from MSSQL",
        description=f"{get_metadata('./database/SQL/')}, if required data is relative, use 'search from MSSQL'.",
        args_schema=SearchSQLInput,
        return_direct=True,
    )

    # === searchResultInput ===
    class SearchResultInput(BaseModel):
        action_input: str = Field(description=f'''
        .
        ''')

    def search_from_result(action_input: str) -> str:

        SummaryFunc.for_sql = 'False'

        return SummaryFunc.query(SummaryFunc.Input_Question)
        
    searchFromResult = StructuredTool.from_function(
        func=search_from_result,
        name="search from Result",
        description=f''' 如果使用者有提到"上述結果中"或是有意圖要透過之前的查詢結果再做更深入的查詢，請使用這個工具。
                        ''',
        args_schema=SearchResultInput,
        return_direct=True,
    )



    # === All Tools ===
    tools = [searchFromOracle, searchFromFile, searchFromSQL, searchFromResult] 


    # === Memory ===
    # from langchain.memory import ConversationBufferWindowMemory
    # from langchain.agents import load_tools

    # memory = ConversationBufferWindowMemory(
    #     memory_key="chat_history", k=2, return_messages=True, output_key="output"
    # )# , output_key="output"

    # === Prompt Template ===
    system = '''
        Respond to the human as helpfully and accurately as possible. You have access to the following tools:

        {tools}

        Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

        Valid "action" values: "Final Answer" or {tool_names}

        Provide only ONE action per $JSON_BLOB, as shown:

        ```
        {{
            "action": $TOOL_NAME,
            "action_input": "String" (Do not use "title" and "description" in the result)
        }}
        ```

        Follow this format:

        Question: input question to answer
        Thought: consider previous and subsequent steps
        Action:
        ```
        $JSON_BLOB
        ```
        Observation: action result
        ... (repeat Thought/Action/Observation N times)
        Thought: I know what to respond
        Action:
        ```
        {{
            "action": "Final Answer",
            "action_input": "Final response to human"
        }}
        ```
        Begin! 
        Use tools if necessary. 
        Respond directly if appropriate. 
        '''
        # Reminder to ALWAYS respond with a valid json blob of a single action. 
        # Format is Action:```$JSON_BLOB```then Observation

    human = '''{input}

        {agent_scratchpad}

        '''
        # (reminder to respond in a JSON Blob without annotations no matter what)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            # MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    )

    # === Agent ===
    my_agent = create_structured_chat_agent(llm, tools, prompt) 
    '''
        create_structured_chat_agent(tools_renderer = my_tools_renderer) 可以修改 tools_renderer

        原 tools_renderer:
            def render_text_description_and_args(tools: List[BaseTool]) -> str:
                tool_strings = []
                for tool in tools:
                    args_schema = str(tool.args)
                    tool_strings.append(f"{tool.name}: {tool.description}, args: {args_schema}")
                return "\n".join(tool_strings)

        prompt = prompt.partial(
        tools=tools_renderer(list(tools)),  # 會被加在 system prompt 的 {tools}
        tool_names=", ".join([t.name for t in tools]),
        )
        
        範例:
        "{tool.name}: {tool.description}, args: {args_schema}"

        "search from Oracle: search from Oracle(Input_Question: str) -> str - if required data is relative, use'search from Oracle'., 
        args: {'Input_Question': {'title': 'Input Question', 'description': '\\n   Expected a Input Question as String. \\n\\n  ', 'type': 'string'}}"

    '''
    # === Agent Executor ===
    return AgentExecutor(agent = my_agent, verbose=True, tools = tools, max_iterations=5, handle_parsing_errors=True) #, handle_parsing_errors=True

Agent_Executor = reload()

if __name__ == "__main__":

    import time

    while True:
        data = input("Message: ")
        if 'Exit' == data:
            break

        SummaryFunc.Input_Question = data

        start = time.time()
        print(Agent_Executor.invoke({"input": data}))
        end = time.time()

        print(f"\nTotal time: {end - start}")
        print("\n\n")
