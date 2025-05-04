import json
import re
import torch
import regex
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# 获取输出
def get_model_output(model,text,tokenizer):
    # 测试输入
    input_text = text
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
          # 生成模型输出
         output_ids = model.generate(
         **model_inputs,
         max_new_tokens=4000,
         do_sample=True,
         top_p=0.95,
         top_k=40
         )
         output_ids = [
               output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)
             ]

         # 解码输出
         output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print("-------- Test Example --------")
    #print(f"Input: {input_text}")
    #print(f"Output: {output_text}")
    return output_text

def normalize_latex(expr):
   #将 \dfrac 替换为 \frac  
    expr = re.sub(r'\\dfrac', r'\\frac', expr) 
    return expr
# 正则化匹配
def extract_boxed_content(s):
    # 定义正则表达式模式，匹配 $\boxed{...}$ 并捕获其中的内容
    pattern = r'\\boxed\{(.*)\}'
    
    # 使用 findall 方法查找所有匹配项
    matches = re.findall(pattern, s)
    
    # 返回匹配项列表
    matches = matches[-1] if len(matches)!=0 else 'NULL'

    return normalize_latex(matches)


def is_exact_match(answer1, answer2):
    return str(answer1) == str(answer2)

def test_model(model_path, prompts, answers):
    model_path = model_path
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path) # 修改为你原模型的地址
    accuracy = []
    ls = zip(prompts, answers)
    f = open("outputqwq32bmath.txt", 'w')
    for prompt, answer in tqdm(ls):
        prompt = "Please reason step by step, and put your final answer within \\boxed{}." + prompt + "<think>\n"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = get_model_output(model, text, tokenizer)
        generated_text = outputs
        output = extract_boxed_content(generated_text)
        if is_exact_match(answer, output):
            accuracy.append(1)
            print("this answer is Correct")
        else:
            accuracy.append(0)
            print("this answer "+ output +" is Wrong")
            print("the correct answer is " + str(answer))
            print("--------------------------------")
        f.write("prompt\n" + prompt + '\n')
        f.write("text\n" + generated_text + '\n')
        f.write("answer\n" + str(answer) + "\n")
        f.write("----------------------------------------------------------\n")
    f.close()    
    print(f"Accuracy: {sum(accuracy) / len(accuracy)}")

if __name__ == "__main__":

    from datasets import load_dataset
    ds_path = "/home/huangmingzhe/scratch/dataset/MATH-500"
    ds = load_dataset(ds_path)  # split=DATASET_SPLIT)
    ds_split = list(ds.keys())[0]
    ds = load_dataset(ds_path, split=ds_split)
    num_sample = len(ds)
    ds = ds.shuffle(seed=42).select(range(num_sample))

    prompts = ds['problem']
    answers = ds['answer']

    test_model(model_path="/home/huangmingzhe/scratch/model/QwQ-32B", prompts=prompts, answers=answers)
