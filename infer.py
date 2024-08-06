import socket

import transformers

print(socket.gethostname())

import inspect
import os
from datetime import datetime

import fire
import huggingface_hub
import torch
from dotenv import load_dotenv
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from peft import PeftModel
from utils.eval_helper import inspectt, logg


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
    adapter_name: str = None,
    # run_id: str = datetime.now().strftime("%y%m%d%H%M%S"),
    resize_token_embeddings: bool = False,
    connect_adapter: bool = False,
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
        # torch_dtype=torch.float16,
        device_map="auto",
    )

    if resize_token_embeddings:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_name, cache_dir=f"{cache_dir}/tokenizer"
        )
        model.resize_token_embeddings(len(tokenizer))
        print("Resized model token embeddings!")

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=f"{cache_dir}/tokenizer"
        )
    if connect_adapter:
        model = PeftModel.from_pretrained(model, adapter_name)
        model = model.merge_and_unload()
        print("Adapter connected!")

    generation_config = {
        "max_new_tokens": 100,
        "do_sample": True,
        # "temperature": 0,
        "top_p": 0.95,
    }

    # better approach if dont have quick access to .generate
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    print(model.generation_config)
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]

    # for inp in prompts:

    instruction = input("Instruction: ")
    while True:
        inp = input("Input: ")
        if inp == "-1":
            break
        if inp == "<-":
            instruction = input("Instruction: ")
            continue
        if inp == "debug":
            import ipdb

            ipdb.set_trace()
            print("debugging")
            continue

        # example = tokenize(instruction, inp, ft_config, tokenizer).to(model.device)
        example = tokenizer(inp, return_tensors="pt").to(model.device)
        output = model.generate(**example, **generation_config)
        response = tokenizer.decode(
            output[0][len(example["input_ids"][0]) :], skip_special_tokens=False
        )

        logg("Response")
        print(response)
        logg("")


if __name__ == "__main__":
    fire.Fire(main)
