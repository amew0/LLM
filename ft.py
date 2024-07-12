import inspect
import os
import gc
import torch
import transformers
import huggingface_hub
import wandb

from utils.eval_helper import inspectt, logg
from utils.ft_helper import generate_and_tokenize_prompt, tokenize

wandb.require("core")
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from time import time

import fire


def main(
    output_dir=f"./out",
    cache_dir=f"/dpc/kunf0097/l3-8b",
    train_data_path="./data/1/medical.json",
    eval_data_path="./data/eval_medical_2k.json",
    eval_split="train",
    log_file=None,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    model_save_path: str = None,
    evaluator_name="meta-llama/Meta-Llama-3-8B-Instruct",
    run_id=datetime.now().strftime("%y%m%d%H%M%S"),
    log2wandb: bool = True,
    project="huggingface",
    entity="my-ku-org",
    evals_per_example=2,
):
    """
    Finetuning.

    Args:

    """

    if model_save_path is None:
        model_save_path = f"{cache_dir}/model/{model_name}-v{run_id}"

    inspectt(inspect.currentframe())

    start = time()
    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    huggingface_hub.login(token=HF_TOKEN_WRITE)
    torch.cuda.empty_cache()

    # Set model and tokenizer paths
    # Initialize tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Initialize model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        return_dict=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/tokenizer",
        padding_side="right",
        pad_token_id=(0),
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for LoRA training
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Tokenize prompt function
    cutoff_len = 256  # most important hp to control CUDA OOM # send to args

    # Load and process dataset
    data = load_dataset("json", data_files=train_data_path)
    train_dataset = (
        data["train"]
        .shuffle()
        .map(lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len))
    )

    # Build Trainer
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,  # !very important to send data to cpu
        warmup_steps=1,
        num_train_epochs=10,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=f"{output_dir}/{run_id}",
        group_by_length=False,
        dataloader_drop_last=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        args=train_args,
        max_seq_length=cutoff_len,
    )

    # Train model
    gc.collect()
    gc.collect()
    trainer.train()

    # Save model and tokenizer
    trainer.model.save_pretrained(model_save_path)

    # saving to load later from https://www.youtube.com/watch?v=Pb_RGAl75VE&ab_channel=DataCamp
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        torch_dtype=torch.float16,
        device_map={"": 0},
        return_dict=True,
    )
    model = PeftModel.from_pretrained(model, model_save_path)
    model = model.merge_and_unload()  # revise it!!

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/tokenizer",
        padding_side="right",
        pad_token_id=(0),
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Push to Hugging Face Hub
    tokenizer.push_to_hub(f"{model_name}-v{run_id}", token=HF_TOKEN_WRITE)
    model.push_to_hub(f"{model_name}-v{run_id}", token=HF_TOKEN_WRITE)

    # Log elapsed time
    end = time()
    logg(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    logg("ft-medical.py")
    fire.Fire(main)
