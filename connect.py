# pushing

import os
from utils.ft_helper import get_start_index
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
HF_TOKEN_WRITE = os.environ["HF_TOKEN_WRITE"]

cache_dir: str = f"/dpc/kunf0097/l3-8b"
train_data_path: str = "meher146/medical_llama3_instruct_dataset"
model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
run_id: str = "240724111548"
last_checkpoint: str = None

# if train_data_path locally exists use it
if os.path.exists(train_data_path):
    data = load_dataset("json", data_files=train_data_path, split="train")
else:
    data = load_dataset(train_data_path, split="train")

if last_checkpoint is not None:
    start_index = get_start_index(last_checkpoint, len(data))

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=f"{cache_dir}/model",
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(last_checkpoint, cache_dir=f"{cache_dir}/tokenizer")
model.resize_token_embeddings(len(tokenizer))


ftmodel = PeftModel.from_pretrained(model, last_checkpoint)

# adapter only
ftmodel.push_to_hub(
    f"{model_name.split('/')[1]}-v{run_id}_si{start_index}-ada", token=HF_TOKEN_WRITE
)
# gotta send the tokenizer as well

ftmodel = ftmodel.merge_and_unload()  #


tokenizer.push_to_hub(
    f"{model_name.split('/')[1]}-v{run_id}_si{start_index}", token=HF_TOKEN_WRITE
)
ftmodel.push_to_hub(
    f"{model_name.split('/')[1]}-v{run_id}_si{start_index}", token=HF_TOKEN_WRITE
)
