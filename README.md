# mlx_ai_agent

## Create virtual environment with pyenv

```sh
pyenv virtualenv 3.12.12 mlx-agent-env
source ~/.pyenv/versions/mlx-agent-env/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn mlx-lm
```

## Download models 

### Download models with huggigface

```sh
export HF_TOKEN=hf_XXXXXX
huggingface-cli download  Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen2.5-7b-instruct
```

### Download models with modelscope

```sh
modelscope download --model Qwen/Qwen-14B-Chat
```

##  LLM server

### Start directly the LLM server with mlx_lm.server

```sh
python -m mlx_lm server --model /Users/zhlsunshine/Projects/inference/models/qwen2.5-14b-instruct-bits-8 --port 8080
```

### Start the LLM server with UI server

```sh
# set the limitation of united memory to 26GBi (26 * 1024 = 26624)
# will be default once the mac restart
sudo sysctl iogpu.wired_limit_mb=28672
pip install fastapi uvicorn mlx-lm
uvicorn app:app --host 0.0.0.0 --port 8000
```

Valid the app service without stream response:

```sh
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "type": "HumanMessage",
        "content": "You are a helpful assistant. What is the process of World War II?"
      }
    ],
    "max_tokens": 128
}'
```

Valid the app service with stream response:

```sh
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "type": "HumanMessage",
        "content": "You are a helpful assistant. What is the process of World War II?"
      }
    ],
    "max_tokens": 512
}'
```

## Start Client to LLM Server

```sh
cd agent
pip install -r requirements.txt
export TAVILY_API_KEY=tvly-dev-******
export SERP_API_KEY=************
python main.py
```