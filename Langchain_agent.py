
import pandas as pd
import duckdb
import time
import re
import os
from langchain_core.prompts import ChatPromptTemplate

from config import Infom

def isAvailable():
    return Infom["Authentication"]["search_csv"]

class getCsvDataset:
    def __init__(self):
        pass
        
    def dataset_query_sql(repo, query):
        
        try:
            if not isAvailable():
                return "您沒有權限"

            files = os.listdir(repo)
            
            for i, file in enumerate(files):
                if file.split(".")[1] != 'csv':
                    continue
                    
                filename = file.split(".")[0]
                # print(filename)
                if filename in query:
                    temp = pd.read_csv(repo+"/"+file, encoding='utf-8')

                    # 更改日期 datatype
                    Dates = []
                    for col in temp.columns:
                        if col in ('TD002', 'MD007', 'SD008', 'DOB005', 'TD003', 'ID003', 'LMD004', 'OD005'):
                            Dates.append(col)
                    temp[Dates] = temp[Dates].apply(lambda x: pd.to_datetime(x, format="%Y%m%d"))
                    # =================

                    locals()[f"table{i}"] = temp
                    query = query.replace(filename, f"table{i}")
                    print(f"table{i} = {filename}\n")
                    
            query = query[query.find('SELECT'): query.find(';')]

            print("final query: \n"+query)

            return duckdb.query(query).df()
        
        except FileNotFoundError as error:
            
            print("檔案不存在")
            return None
    

if __name__ == "__main__":
    
    pass
    
        
            
    
