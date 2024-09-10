import neptune
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

run = neptune.init_run(
    project="xuchi/InferenceTimeAnalyze",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5Zjc5NDhlNS04MmE5LTQzOTctOWNkOC0yMGY4YWNmNGRmNjUifQ==",
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(1)
device = torch.device("cuda:1")

# 加载模型和tokenizer
for i in range(10):
    model_name = "/home/workingplace/xc/exercise/Relative-importance-and-activation-pruning/llm_weights_"+str(i+1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    device = model.hf_device_map["lm_head"]
    # 准备输入数据
    input_text = "The quick brown fox"
    inputs = tokenizer(input_text, return_tensors='pt').to(device)  # 使用GPU
    inputs.to(device)
    # 测量推理时间
    num_trials = 500
    total_time = 0

    start_time = time.time()
    for _ in range(num_trials):
        with torch.no_grad():
            outputs = model.generate(**inputs)
    end_time = time.time()
    total_time = end_time - start_time

    average_inference_time = total_time / num_trials
    run["total_inf_time"] = total_time
    run["inf_time/total_inf_time"].append(total_time)

    if i>2:
        print(f"Total Inference time over {num_trials} trials: {total_time:.4f} seconds for {i-2}:8 sparsity")
        continue
    print(f"Total Inference time over {num_trials} trials: {total_time:.4f} seconds for {i+1}:4 sparsity")

