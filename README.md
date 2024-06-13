# Fubon

## NodeJS Prebulit Binaries
1. 下載並解壓縮 [NodeJS 20.XX.X LTS](https://nodejs.org/en/download/current)
![image](https://github.com/valentine620721/BranchAssistant/assets/48355136/9bd680a3-8a6c-42f3-a89d-33f23d72229b)
2. 將 node 資料夾的路徑加入環境變數
![image](https://github.com/valentine620721/BranchAssistant/assets/48355136/152819dc-d2a8-4080-aab0-7a4cea82ed05)
3. 進入專案資料夾中的 client 資料夾
```bash
cd .../client
npm install
```
## Python 安裝

1. 下載並安裝 [Python 3.12.2](https://www.python.org/downloads/release/python-3122/)

2. 
用 [pip](https://pip.pypa.io/en/stable/) 安裝所有需要的套件 
```bash
pip install -r requirements.txt
```
## Ollama 安裝
1. 下載並安裝 [Ollama](https://ollama.com/download)
2. 使用 Ollama
```bash
Ollama run mistral or Ollama run phi3 or Ollama run qwen2:7b
## 模型:參數量 ，不加參數代表預設參數版本，各模型預設版本可由Ollama Library查詢
Ollama pull shaw/dmeta-embedding-zh
```
3. 可以從 [Ollama library](https://ollama.com/library) 找到可以用的模型

註: 模型下載完畢，若進入管理介面，可以輸入 /bye退出。
## 運行程式
1. 確定 Ollama 正在運行
```bash
ollama serve

# 跳出 Error 就是正在運行
Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```
2. 進入專案資料夾後執行後端程式

```bash
python app.py
```

3. 進入專案資料夾的 client 後執行前端程式

```bash
cd .../Project/client
npm start
```
