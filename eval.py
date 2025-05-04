
from evalscope import TaskConfig, run_task

task_config = TaskConfig(
    model='/home/huangmingzhe/scratch/model/DeepSeek-R1-Distill-Qwen-7B',  #模型名称 
    datasets=['aime24'],  #数据集名称
    dataset_args={'aime24': {'few_shot_num': 0,'local_path':'/home/huangmingzhe/scratch/dataset/AIME_2024'}},
    generation_config={
        'max_tokens': 30000,  #最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.6,  #采样温度 (qwen 报告推荐值)
        'top_p': 0.95,  #top-p采样 (qwen 报告推荐值)
        'top_k': 40,  #top-k采样 (qwen 报告推荐值)
        'n': 1,  #每个请求产生的回复数量
    },  
    model_args={'device_map': 'sequential'}
)
run_task(task_config)

