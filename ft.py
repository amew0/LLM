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

wandb.require("core")
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


def main(
    cache_dir: str = f"/dpc/kunf0097/l3-8b",
    # train_data_path: str = "./data/medical.json",
    train_data_path: str = "meher146/medical_llama3_instruct_dataset",
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    model_save_path: str = None,
    run_id: str = datetime.now().strftime("%y%m%d%H%M%S"),
    chpt_dir: str = None,
    last_checkpoint: str = None,
    start_index: int = 0,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    world_size: int = None,
    local_rank: int = None,
):
    """
    Finetuning.

    Args:
        cache_dir (str): Directory for caching models.
        train_data_path (str): Path to training data.
        model_name (str): Name of the model to fine-tune.
        model_save_path (str): Path to save the fine-tuned model.
        run_id (str): Unique identifier for the run.
        chpt_dir (str): Directory for checkpoints.
        last_checkpoint (str): Path to the last checkpoint.
        start_index (int): Start index for the dataset.
        per_device_train_batch_size (int): Batch size per device.
        gradient_accumulation_steps (int): Steps for gradient accumulation.
        world_size (int): Number of distributed processes.
        local_rank (int): Local rank for distributed training.
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

    # if train_data_path locally exists use it
    if os.path.exists(train_data_path):
        data = load_dataset("json", data_files=train_data_path, split="train")
    else:
        data = load_dataset(train_data_path, split="train")

    if last_checkpoint is not None:
        start_index = get_start_index(last_checkpoint, len(data))

    # device_map = "auto"
    device_map = {"": 0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    ddp = world_size != 1
    if ddp:
        device_map = {"": local_rank}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    inspectt(inspect.currentframe())
    logger = logging.getLogger(__name__)

    start = time()
    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    huggingface_hub.login(token=HF_TOKEN_WRITE)
    torch.cuda.empty_cache()

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
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"{cache_dir}/tokenizer")
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
    cutoff_len = 248  # (75% of the data wont be affected)

    # Process dataset
    if start_index != 0:
        data = reorder_dataset(data, start_index)
    train_dataset = data.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, cutoff_len))

    # load it from .yaml
    train_args = SFTConfig(
        run_name=f"ft-{model_name.split('/')[1]}-{run_id}-v{start_index}",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=1,  # !very important to send data to cpu
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=3e-4,
        fp16=False,
        logging_steps=10,
        optim="adamw_torch",
        output_dir=f"{chpt_dir}",
        group_by_length=False,
        dataloader_drop_last=False,
        save_steps=400,
        save_total_limit=3,
        max_seq_length=cutoff_len, # not sure its purpose since its setup on the tokenizer
        resume_from_checkpoint=last_checkpoint,
        load_best_model_at_end=True, # experimental
    )

    class SFTTrainerNoShuffle(SFTTrainer):
        def training_step(self, model, inputs):
            if (self.state.global_step % self.args.save_steps) == 0:
                inputs_decoded = tokenizer.decode(inputs["input_ids"][0])
                logger.info(f"{self.state.global_step}: {inputs_decoded}")
            return super().training_step(model, inputs)

        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)  # to prevent shuffling

        # ddnt try it yet
        def save_state(self):
            self.state.gradient_accumulation_steps = self.args.gradient_accumulation_steps
            super().save_state()

    trainer = SFTTrainerNoShuffle(
        model=model,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=True
        ),
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

    # Log elapsed time
    end = time()
    logg(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    logg("ft-medical.py")
    fire.Fire(main)
