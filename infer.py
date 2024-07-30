import socket

print(socket.gethostname())

import gc
import inspect
import os
import logging
from datetime import datetime
from time import time

import fire
import huggingface_hub
import torch
import transformers
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

from peft import LoraConfig, PeftModel
from utils.eval_helper import inspectt, logg
from utils.ft_helper import (
    generate_and_tokenize_prompt,
    get_start_index,
    reorder_dataset,
)
from torch.utils.data import SequentialSampler

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()
wandb.require("core")


def tokenize(instruction, inp, ft_config, tokenizer, verbose=True):
    user_prompt = ft_config["prompt"].format(instruction, inp)
    if verbose:
        logg("User Prompt")
        print(user_prompt)
        logg("")
    tokenized = tokenizer(user_prompt, return_tensors="pt")
    return tokenized


def main(
    cache_dir: str = f"/dpc/kunf0097/l3-8b",
    model_name: str = "EleutherAI/pythia-70m-deduped",
    adapter_name: str = "amew0/pythia-70m-deduped-v240730153003_si114138-ada",
    run_id: str = datetime.now().strftime("%y%m%d%H%M%S"),
    resize_token_embeddings: bool = True
):
    """
    Inference.

    Args:
    """
    load_dotenv()

    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    if HF_TOKEN_WRITE is not None:
        huggingface_hub.login(token=HF_TOKEN_WRITE)

    inspectt(inspect.currentframe())

    with open(f"tuning.yaml", "r") as f:
        ft_config = yaml.safe_load(f)[model_name]
        assert "prompt" in ft_config, "Prompt template is not defined in tuning.yaml"
        print(ft_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_name, cache_dir=f"{cache_dir}/tokenizer")
    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    ftmodel = PeftModel.from_pretrained(model, adapter_name)
    ftmodel = ftmodel.merge_and_unload()


    generation_config = {
        "max_new_tokens": 200
    }

    instruction = input("Instruction: ")
    while True:
        inp = input("Input: ")
        if inp == "-1":
            break
        if inp == "<-":
            instruction = input("Instruction: ")
            inp = input("Input: ")


        example = tokenize(instruction, inp, ft_config, tokenizer).to(model.device)

        output = model.generate(**example, **generation_config)
        response_ids = output[0][len(example["input_ids"][0]) :]
        response = tokenizer.decode(response_ids, skip_special_tokens=False)

        logg("Response")
        print(response)
        logg("")


if __name__ == "__main__":
    fire.Fire(main)
