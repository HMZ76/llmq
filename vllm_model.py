from datasets import load_dataset 
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
import re

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def extract_boxed_content(s):
    # 定义正则表达式模式，匹配 $\boxed{...}$ 并捕获其中的字符
    pattern = r'\\boxed\{(.*?)\}'
    
    # 使用 findall 方法查找所有匹配项，非贪婪匹配
    matches = re.findall(pattern, s)
    print(matches)
    # 返回匹配项列表
    return matches[-1] if len(matches)!=0 else "NULL"



def get_completion(prompts, model, tokenizer=None, max_tokens=32000, temperature=0.6, top_p=0.95, top_k=40, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率,避免无休止的重复
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, top_k=top_k, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True,tensor_parallel_size=2)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    #model='../model/QwQ-32B' # 指定模型路径
    model='/home/huangmingzhe/scratch/model/DeepSeek-R1-Distill-Qwen-32B'#-W8A8-Dynamic-Per-Token'
    # model='/root/autodl-tmp/Qwen/QwQ-32B-AWQ' # 指定模型名称，自动加载模型
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    #tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) 
    
    # Select calibration dataset.
    DATASET_ID = "/home/huangmingzhe/scratch/dataset/AIME_2024"
    DATASET_SPLIT = "train"
   
    NUM_CALIBRATION_SAMPLES = 30  
    MAX_SEQUENCE_LENGTH = 2048

     
    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {
              "text":"Please reason step by step, and put your final answer within \\boxed{}." + example['Problem'] + "<think>\n"
         }   

    ds = ds.map(preprocess)   
    answer = ds['Answer'][0:3]
    text = [d['text'] for d in ds][0:3]
    #text = [tokenizer.apply_chat_template(t) for t in text]

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=8192, temperature=0.6, top_p=0.95, top_k=40, max_model_len=2048) # 思考需要输出更多的 Token 数，max_tokens 设为 8K，根据 qwen 官方的建议，temperature应在 0.5-0.7，推荐 0.6

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    cnt = 0
    acc = 0
    c=0
    for output in outputs:
        c=c+1
        if c > 2:
           break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if r"<think>" in generated_text:
            think_content, answer_content = generated_text.split(r"<think>")
        else:
            think_content = ""
            answer_content = generated_text
         
        answer_content = answer_content.replace(r"\(",'$').replace(r"\)","$")
        ans = extract_boxed_content(answer_content)
        if str(ans) == str(answer[cnt]):
           acc += 1
        print(ans)
        cnt += 1
        
    print(acc/cnt)
            
