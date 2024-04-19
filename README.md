# Fubon


## Python 安裝

1. 下載並安裝 [Python 3.12.2](https://www.python.org/downloads/release/python-3122/)

2. 用 [pip](https://pip.pypa.io/en/stable/) 安裝所有需要的套件 

```bash
pip install -r requirements.txt
```
## Ollama 安裝
1. 下載並安裝 [Ollama](https://ollama.com/download)
2. 使用 Ollama
```bash
Ollama run mistral
```
3. 可以從 [Ollama library](https://ollama.com/library) 找到可以用的模型


## 運行程式
1. 確定 Ollama 正在運行
```bash
ollama serve

# 跳出 Error 就是正在運行
Error: listen tcp 127.0.0.1:11434: bind: Only one usage of each socket address (protocol/network address/port) is normally permitted.
```
3. 進入專案資料夾後執行

```bash
python app.py
```
