import gc
import inspect
import os
from datetime import datetime
from time import time
import yaml

import fire
import huggingface_hub
import torch
import transformers
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from peft import LoraConfig, PeftModel
from utils.eval_helper import inspectt, logg
from utils.ft_helper import (
    generate_and_tokenize_prompt,
    reorder_dataset,
)
from torch.utils.data import DataLoader, SequentialSampler

import coloredlogs

import logging

logging.basicConfig(level=logging.INFO)


wandb.require("core")


def main(
    output_dir=f"./out",
    cache_dir=f"/dpc/kunf0097/l3-8b",
    train_data_path="./data/medical.json",
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    model_save_path: str = None,
    run_id=datetime.now().strftime("%y%m%d%H%M%S"),
    chpt_dir: str = None,
    last_checkpoint=None,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    prompt_template=None
):
    """
    Finetuning.

    Args:

    """

    if model_save_path is None:
        model_save_path = f"{cache_dir}/model/{model_name}-v{run_id}"

    if chpt_dir is None:
        chpt_dir = f"{cache_dir}/chpt/{run_id}"

    if os.path.isdir(chpt_dir):
        checkpoints = [d for d in os.listdir(chpt_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(
                chpt_dir, max(checkpoints, key=lambda cp: int(cp.split("-")[-1]))
            )

    start_index = 0
    if last_checkpoint is not None:
        start_index = (
            int(last_checkpoint.split("-")[-1])
            * per_device_train_batch_size
            * gradient_accumulation_steps
        )

    if prompt_template is None:
        with open("tuning.yaml", "r") as f:
            tuning_config = yaml.safe_load(f)
            prompt_template = tuning_config[model_name]["prompt_template"]


    inspectt(inspect.currentframe())
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG", logger=logger)

    start = time()
    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    huggingface_hub.login(token=HF_TOKEN_WRITE)
    torch.cuda.empty_cache()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"{cache_dir}/tokenizer")

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
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # Prepare model for LoRA training
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # most important hp to control CUDA OOM # send to args
    cutoff_len = 296  # (75% of the data wont be affected)

    # Load and process dataset
    # data = load_dataset("json", data_files=train_data_path, split="train")
    data = load_dataset("meher146/medical_llama3_instruct_dataset", split="train")
    if start_index != 0:
        data = reorder_dataset(data, start_index)
    train_dataset = data.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len, prompt_template))

    # load it from .yaml
    train_args = SFTConfig(
        run_name=f"ft-{model_name.split('/')[1]}-{run_id}-v{start_index}",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # only 1 is allowed on the no shuffler [needs revision]
        eval_accumulation_steps=1,  # !very important to send data to cpu
        warmup_steps=1,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=1,
        optim="adamw_torch",
        output_dir=f"{chpt_dir}",
        group_by_length=False,
        dataloader_drop_last=False,
        save_steps=400,
        save_total_limit=3,
        max_seq_length=cutoff_len,
        resume_from_checkpoint=last_checkpoint,
    )

    class SFTTrainerNoShuffle(SFTTrainer):
        def training_step(self, model, inputs):
            if (self.state.global_step % self.args.save_steps) == 0:
                inputs_decoded = tokenizer.decode(inputs["input_ids"][0])
                logger.info(f"{self.state.global_step}: {inputs_decoded}")
            return super().training_step(model, inputs)

        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)  # to prevent shuffling

    trainer = SFTTrainerNoShuffle(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        args=train_args,
    )

    # Train model
    gc.collect()
    gc.collect()

    if last_checkpoint is not None:
        trainer.train(last_checkpoint)
    else:
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
