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
    GenerationConfig,
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
# wandb.require("core")


def main(
    cache_dir: str = f"/dpc/kunf0097/l3-8b",
    train_data_path: str = "meher146/medical_llama3_instruct_dataset",
    model_name: str = "EleutherAI/pythia-70m-deduped",
    model_save_path: str = None,
    run_id: str = datetime.now().strftime("%y%m%d%H%M%S"),
    chpt_dir: str = None,
    last_checkpoint: str = None,
    start_index: int = 0,
    cutoff_len: int = 298,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    world_size: int = None,
    local_rank: int = None,
):
    """
    Finetuning.

    Args:
        cache_dir (str): Directory for caching models/tokenizers/datasets.
        train_data_path (str): Path to training data.
        model_name (str): Name of the model to fine-tune.
        model_save_path (str): Path to save the fine-tuned model.
        run_id (str): Unique identifier for the run.
        chpt_dir (str): Directory for checkpoints.
        last_checkpoint (str): Path to the last checkpoint.
        start_index (int): Start index for the dataset.
        cutoff_len (int): Cutoff length for the dataset (For batching).
        per_device_train_batch_size (int): Batch size per device.
        gradient_accumulation_steps (int): Steps for gradient accumulation.
        world_size (int): Number of distributed processes.
        local_rank (int): Local rank for distributed training.
    """
    load_dotenv()
    logger = logging.getLogger(__name__)

    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    if HF_TOKEN_WRITE is not None:
        huggingface_hub.login(token=HF_TOKEN_WRITE)

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
        data = load_dataset(
            "json",
            data_files=train_data_path,
            split="train",
            cache_dir=f"{cache_dir}/datasets",
        )
    else:
        data = load_dataset(
            train_data_path,
            split="train",
            cache_dir=f"{cache_dir}/datasets",
        )

    if last_checkpoint is not None:
        start_index = get_start_index(last_checkpoint, len(data))

    device_map = {"": 0}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    ddp = world_size != 1
    if ddp:
        device_map = {"": local_rank}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    inspectt(inspect.currentframe())

    with open(f"tuning.yaml", "r") as f:
        ft_config = yaml.safe_load(f)[model_name]
        # assert "training_args" in ft_config, "training_args aren't defined in tuning.yaml"
        # assert "peft_args" in ft_config, "Peft arguments are not defined in tuning.yaml"
        print(ft_config)

    # Initialize model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    torch_dtype = torch.float16
    if not torch.cuda.is_available():
        bnb_config = None
        device_map = "cpu"
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    generation_config = ft_config["generation_config"]
    # better approach if dont have quick access to .generate
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    print(model.generation_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f"{cache_dir}/tokenizer")
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad token. Adding it.")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    # import sys;sys.exit(0)

    # Prepare LoRA
    peft_args = ft_config["peft_args"]
    peft_config = LoraConfig(**peft_args)

    # cutoff_len most important hp to control CUDA OOM # send to args
    # (75% of the data wont be affected by 298 - Q3)

    if start_index != 0:
        data = reorder_dataset(data, start_index)
    train_dataset = data.map(
        lambda x: generate_and_tokenize_prompt(x, tokenizer, ft_config, cutoff_len)
    )

    training_args = ft_config["training_args"]
    train_config = SFTConfig(
        run_name=f"ft-{model_name.split('/')[1]}-{run_id}-v{start_index}",
        resume_from_checkpoint=last_checkpoint,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=f"{chpt_dir}",
        max_seq_length=cutoff_len,  # not sure its purpose since its setup on the tokenizer
        eval_accumulation_steps=1,  # !very important to send data to cpu
        **training_args,
    )

    class SFTTrainerNoShuffle(SFTTrainer):
        def training_step(self, model, inputs):
            if (self.state.global_step % self.args.save_steps) == 0:
                inputs_decoded = tokenizer.decode(inputs["input_ids"][0])
                logger.critical(f"Step {self.state.global_step}\n{inputs_decoded}")
            return super().training_step(model, inputs)

        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)  # to prevent shuffling

    trainer = SFTTrainerNoShuffle(
        model=model,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, return_tensors="pt", padding=True
        ),
        peft_config=peft_config,
        train_dataset=train_dataset,
        args=train_config,
    )

    max_steps = (len(train_dataset) * trainer.args.num_train_epochs) / (
        per_device_train_batch_size * gradient_accumulation_steps * world_size
    )

    logg(f"Total steps: {max_steps}")
    # Train model
    gc.collect()
    gc.collect()

    start = time()
    if last_checkpoint is not None:
        logg("Resuming from checkpoint")
        trainer.train(last_checkpoint)
    else:
        trainer.train()
    end = time()
    logg(f"Elapsed time: {end - start}")

    trainer.model.save_pretrained(model_save_path)


if __name__ == "__main__":
    logg("ft-medical.py")
    fire.Fire(main)
