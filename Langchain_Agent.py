
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

import difflib
import re

# === LLM ===
from langchain_community.llms import Ollama

# llm = Ollama(model = "llama3", temperature = 0, num_predict=320)
# llm2 = Ollama(model = "gemma:2b", temperature = 0.2)
llm = Ollama(model = "gemma2", top_k=1) #llama3-chatqa:8b, qwen:4b(Alibaba), aya:8b(Cohere), phi3 (3.8B)(Microsoft), gemma:2b zhaoyun/phi3-128k-chinese

SummaryFunc = getSummary(llm) # 宣告一個儲存資料和做總結的物件
RetrievalQA = getFAISS(llm)


# def get_fields(directory):    
#     with open(directory+"tables.txt", 'r') as file:
#         result = file.read()
    
#     return result

def get_fields(lines, table_name):    
    result = ''
    columns_list = []
    for field in lines[1:]:
        if ((table_name.strip()) in field.split('|')[0]):
            result += (field.split('.')[-1]+'\n')
            columns_list.append(field.split('|')[0].split('.')[-1].strip())

    return result, columns_list


def get_metadata(directory):    
    with open(directory+"table_meta.txt", 'r', encoding='utf-8') as file:
        result = file.read()
    
    return result

def get_tables_name(directory):
    name_list = []
    for csv in os.listdir(directory):
        csv_name, csv_extentsion = csv.split('.')
        if  csv_extentsion == "csv":
            name_list.append(csv_name)
    
    return ", ".join(name for name in name_list)

def extract_select_columns(sql):
    select_columns = []
    pattern = r'SELECT(.*?)FROM'
    for columns_string in re.findall(pattern, sql):
        columns_list = columns_string.split(',')
        
        for column in columns_list:
            match = re.search(r'\((.*?)\)', column)
            if match:
                select_columns.append(match.group(1).strip().split('.')[-1])
            else:
                select_columns.append(column.strip().split('.')[-1])
    return set(select_columns)

def get_equal_rate(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

def return_top_similarity(wrong_column, compared_columns_list, topk=3):
    ''' 
        result:
            [('CUS001', 0.8333333333333334),
            ('CN002', 0.7272727272727273),
            ('TD002', 0.5454545454545454)]
    '''

    similarity = {}
    for compared_column in compared_columns_list:
        similarity[compared_column] = get_equal_rate(wrong_column, compared_column)

    return sorted(similarity.items(), key=lambda item: item[1], reverse=True)[:topk]

    

def parse_question(msg):
    system = f'''
        parse question into several key points, then answer as format in chinese.
        Ex:
        Q. 告訴我2025/04/23到期的債券類型產品。 包含產品名稱及到期日。
        A.
        資料條件：
        1. 2025/04/23到期
        2. 債券類型
        3. 產品
        資料欄位：
        1. 產品名稱
        2. 到期日

        Q. 告訴我2021年全年銷售總額第二高的基金類型的產品。 包含產品名稱與總銷售金額。
        A.
        資料條件：
        1. 2021年
        2. 全年銷售總額
        3. 第二高
        4. 基金類型
        5. 產品
        資料欄位：
        1. 產品名稱
        2. 總銷售金額

        Q. 告訴我Email為tracy35@example.org的客戶最近一次購買的產品，包含產品名稱與購買日期。
        A.
        資料條件：
        1. Email為tracy35@example.org
        2. 客戶
        3. 最近一次
        4. 購買的產品
        資料欄位：
        1. 產品名稱
        2. 購買日期

        Q. 告訴我所有出生日期在1990年前的員工的總業績金額。包含員工姓名與業績總額。
        A.
        資料條件：
        1. 所有
        2. 出生日期在1990年前
        3. 員工
        4. 總業績金額
        資料欄位：
        1. 員工姓名
        2. 業績總額

        Q. 告訴我所有年齡大於50歲客戶的交易記錄，包含客戶姓名與每筆交易金額。
        A.
        資料條件：
        1. 所有
        2. 年齡大於50歲
        3. 客戶
        4. 交易記錄
        資料欄位：
        1. 客戶姓名
        2. 每筆交易金額

        Q. 告訴我員工賴芯菡已完成的交易筆數。包含員工姓名。
        A.
        資料條件：
        1. 員工賴芯菡
        2. 已完成
        3. 交易筆數
        資料欄位：
        1. 員工姓名
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
        
        List all 'table_name' that is related to question.
        Ex: table_name_1, table_name_2
        No more explanations or messages, only 'table_name' and split by ','.
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

    First, follow the example:
    1. 問題中有與'總數'、'總'相關字詞： SELECT SUM() FROM table_name GROUP BY
    2. 問題中有與'筆數'相關字詞: SELECT COUNT() GROUP BY
    3. 問題需要不只一個 table 的欄位： JOIN ON fields with diferent name in different tables
    4. 問題中有與日期'年'、'月'相關字詞： YEAR() or MONTH()
    5. 問題中有與'最'、'第n高'相關字詞： ORDER BY field_name (DESC or ASC) LIMIT 1 (OFFSET n-1);
    6. 字串匹配優先使用： WHERE field_name LIKE '%STRING%'

    Then, follow the rules:
    1. use HAVING aggregates condition instead of WHERE.
    2. JOIN all table that need to be queried.

    Finally, answer as format 
    ```sql
        SELECT (DISTINCT) field_name FROM table_name;
    ```, choose field_name depends on metadata.
    '''

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
    tables = generate_tables_name(get_metadata(SummaryFunc.database), SummaryFunc.Input_Question, llm) # Ollama(model = model2, top_k=1)
    # print(f'Tables: {tables}')
    
    with open(SummaryFunc.database+"col_meta.txt", 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')

    table_inform = f'table_name: {tables}\n'
    table_columns_list = []
    # table_inform = ""

    for table_name in tables.split(','):
        # print(f"--table--: {table_name}\n")
        table_inform += f"{lines[0].replace('table_name', table_name.strip())}\n"
        if table_name.strip() != '':
            temp_inform, temp_columns_list = get_fields(lines, table_name)
            table_inform += temp_inform + "\n"
            table_columns_list += temp_columns_list

    print(table_inform)
    print("=====table_columns_list=====")
    print(table_columns_list)

    sqlCommand = generate_SQL(table_inform, SummaryFunc.Input_Question, llm)  # Ollama(model = model3, top_k=1)
    sqlCommand = sqlCommand[sqlCommand.find('SELECT'): sqlCommand.find(';')]

    # 資料檢核改成  透過程式判斷欄位不存在時，用 LLM + 問題+不存在的欄位資訊做修正
    select_columns = extract_select_columns(sqlCommand)
    for column in select_columns:
        if column not in table_columns_list:
            candidate_columns = return_top_similarity(column, table_columns_list)
            print(candidate_columns)

            sqlCommand = sqlCommand.replace(column, candidate_columns[0][0])

    SummaryFunc.sqlCommand = sqlCommand

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

        return parse_question(SummaryFunc.Input_Question) 

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

        return parse_question(SummaryFunc.Input_Question)
        

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
