# vllm_openai_completions.py
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="123", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="Qwen2.5-1.5B-Instruct",
  messages=[
    {"role": "user", "content": "10的阶乘是多少？<think>\n"}
  ]
)
print(completion.choices[0].message)
