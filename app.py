from flask import Flask, request, render_template, make_response, jsonify, send_file
from flask_cors import CORS, cross_origin
from getCustomerDataset import  getCsvDataset
import time
import os
import tempfile
import pandas as pd
import json

import io
from urllib.parse import quote

from Langchain_Agent import Agent_Executor, SummaryFunc, RetrievalQA, llm, reload, query_by_SQL


# Global Variables: 儲存資料來渲染表格
Ai_response = ""

# Flask App
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

# Backend API

# 測試用
@app.route('/get_test', methods=['POST', 'OPTIONS'])
@cross_origin()
def test_input():
    
    header={
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST'
    }

    sqlCommand = '''
        SELECT *
        FROM 客戶交易資料表
    '''
    data = request.get_json()
    SummaryFunc.Input_Question = data.get('inputText')
    result = SummaryFunc.store_dataset(getCsvDataset.dataset_query_sql("./database/Oracle", sqlCommand))

    return make_response(jsonify({'response': "測試", 'data': result.to_json(orient='records')}), 200, header)

@app.route('/get_summary_test', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_summary_test():
    
    header={
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST'
    }

    try:
        data = request.get_json()
        user_input = data.get('inputText')
        SummaryFunc.Input_Question = user_input
        with open(SummaryFunc.test_folder + SummaryFunc.test_jsonfile, 'r') as f:
            data = json.load(f)
            if user_input in data.keys():
                temp = pd.read_csv(SummaryFunc.test_folder + data[user_input], encoding='utf-8')
                SummaryFunc.store_dataset(temp, True)

                start = time.time()
                result = SummaryFunc.summary()
                end = time.time()

                print(f"\nTotal Summary time: {end - start}")
            else:
                result = "沒有這個查詢結果"

    except Exception as e:
        result = "我不知道怎麼回答"
        print(f'\nThe error message is here: {e}')

    return make_response(jsonify({'response': result}), 200, header)

# SQL
@app.route('/get_answer', methods=['POST', 'OPTIONS'])
@cross_origin()
def process_input():

    global Ai_response

    header={
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST'
    }

    data = request.get_json()
    print(f'server receive data from User input: {data}')
    user_input = data.get('inputText') #等同於 data['inputText']
    try:
        SummaryFunc.Input_Question = user_input

        start = time.time()
        Ai_response = Agent_Executor.invoke({"input": user_input})['output']
        end = time.time()

        print(f"\nTotal Query time: {end - start}")
        
        
        if not isinstance(Ai_response, str):
            if (Ai_response is None or len(Ai_response) == 0):
                result = '資料不存在'

            result = "完成資料搜尋"
        else:
            result = Ai_response

        return make_response(jsonify({'response': result, 'check': SummaryFunc.for_sql}), 200, header)

    except Exception as e:
        result = "我不知道怎麼回答"
        print(f'\nThe error message is here: {e}')
    
    return make_response(jsonify({'response': result, 'check': 'False'}), 200, header)
    

@app.route("/get_query", methods=['POST', 'OPTIONS'])
@cross_origin()
def get_query():

    header={
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST'
    }

    data = request.get_json()
    print(f'server receive data from User input: {data}')
    user_input = data.get('inputText') #等同於 data['inputText']

    if(user_input != '1'):

        result = "請重新詢問"

        return make_response(jsonify({'response': result, 'check': 'False'}), 200, header)

    try:
        Ai_response = query_by_SQL()

        if not isinstance(Ai_response, str):
            if (Ai_response is None or len(Ai_response) == 0):
                result = '資料不存在'

            result = "完成資料搜尋"
        else:
            result = Ai_response

        return make_response(jsonify({'response': result, 'check': 'False'}), 200, header)

    except Exception as e:
        result = "我不知道怎麼回答"
        print(f'\nThe error message is here: {e}')
    
    return make_response(jsonify({'response': result, 'check': 'False'}), 200, header)



# 表格資料回傳
@app.route("/table")
def table():
    id = int(request.args.get('id'))
    if id > len(SummaryFunc.datasets)-1:
         id = len(SummaryFunc.datasets)-1

    try:
        header={
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET'
        }
        if SummaryFunc.datasets[id].empty:
            print(SummaryFunc.datasets[id])
            return make_response(jsonify([{"question":SummaryFunc.history[id], "data":[{"無欄位名稱": "無欄位資料"}], "count": SummaryFunc.count}]), 200, header)

        else:
            df = SummaryFunc.datasets[id].copy(deep=True)
            for column in df.dtypes[df.dtypes == 'datetime64[ns]'].index:
                df[column] = df[column].apply(lambda x: x.strftime("%Y-%m-%d"))

            json_data = df.to_json(orient='records')
            parsed = json.loads(json_data)

            return make_response(jsonify([{"question":SummaryFunc.history[id], "data": json_data, "count": SummaryFunc.count}]), 200, header)
    except Exception as e:
        print(str(e)+"\n")
        return make_response(jsonify([{"question":"無查詢紀錄", "data":[{"無欄位名稱": "無欄位資料"}], "count": SummaryFunc.count}]), 200, header)


@app.route("/download")
def download():
    id = int(request.args.get('id'))
    if id > len(SummaryFunc.datasets)-1:
         id = len(SummaryFunc.datasets)-1

    id_name = id+1
    
    if SummaryFunc.count > 5:
        id_name += (SummaryFunc.count-5)

        
    print(f"saved length: {len(SummaryFunc.datasets)}")
    print(f"id: {id}")
    
    header={
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET'
            }

    try:
        if SummaryFunc.datasets[id].empty:
            print(SummaryFunc.datasets[id])

            return make_response("無資料可下載", 404, header)

        else:
            header={
                'Content-Type': "text/csv",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET',
                "Content-Disposition": "attachment; filename*=UTF-8''{utf_filename}".format(utf_filename=quote(f"第{id_name}個查詢結果.csv".encode('utf-8')))
            }

            df = SummaryFunc.datasets[id].copy(deep=True)
            for column in df.dtypes[df.dtypes == 'datetime64[ns]'].index:
                df[column] = df[column].apply(lambda x: x.strftime("%Y-%m-%d"))

            print("準備下載...")
            
            return make_response(df.to_csv(encoding='utf-8', index=False), 200, header)
            # buffer = io.BytesIO()
            # df.to_csv(buffer,encoding='utf-8')
            # buffer.seek(0)
            # return send_file(buffer,
            #      download_name="test.csv",
            #      mimetype='text/csv')

    except Exception as e:
        print(str(e)+"\n")
        return make_response("錯誤", 404, header)

# 彙總
@app.route('/get_summary')
def get_summary():
    
    header={
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'
    }

    try:
        # result = SummaryFunc.query(user_input)
        start = time.time()
        # result = SummaryFunc.summary()
        result = "總結測試"
        end = time.time()

        print(f"\nTotal Summary time: {end - start}")

    except Exception as e:
        result = "我不知道怎麼回答"
        print(f'\nThe error message is here: {e}')

    return make_response(jsonify({'response': result}), 200, header)


# 上傳檔案
@app.route('/upload_doc', methods=['POST'])
@cross_origin()
def upload_documents():
    responses = []
    print(f'\nfile objects that I receive in this upload:{request.files}')
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request.'}), 400

    files = request.files.getlist('file')  # Get the list of files

    for file in files:
        print('\nthis file:', file)
        if file.filename == '':
            responses.append({'message': '''檔案不存在''', 'code': 400})
        
        try:
            saved_file_name = "./database/File/" + file.filename

            if os.path.isfile(saved_file_name):
                responses.append({
                    'message': f'{file.filename} 檔案已存在', 'code': 200
                })
            else:    
                file.save(saved_file_name)
                RetrievalQA.process_and_store_documents([saved_file_name])
                responses.append({
                    'message': f'{file.filename} 上傳成功', 'code': 200
                })
            
        except Exception as e:
            app.logger.debug('Debug message')
            print(str(e))
            responses.append({
                    'message': f'{file.filename} 上傳失敗', 'code': 200
                })

    return jsonify({'responses': responses}), 200


# RAG
@app.route('/get_qa', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_qa():
    
    header={
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST'
    }
    data = request.get_json()
    print(f'server receive data from User input: {data}')
    user_input = data.get('inputText') #等同於 data['inputText']

    try:
        result = RetrievalQA.invoke(user_input)

    except Exception as e:
        result = "我不知道怎麼回答"
        print(f'\nThe error message is here: {e}')

    return make_response(jsonify({'response': result}), 200, header)

@app.after_request
def after_request(response):
    response.access_control_allow_origin = "*"
    return response

if __name__ == "__main__":
    app.run(debug=True, port=5001)
