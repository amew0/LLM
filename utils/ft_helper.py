import os
import json
from datasets import concatenate_datasets
from transformers import TrainerCallback
import torch

from utils.eval_helper import logg


def get_start_index(last_checkpoint, total_rows) -> int:
    """
    Get the start index to resume training from the last checkpoint.

    Args:
        last_checkpoint: str
        total_rows: int
    Returns:
        start_index: int
    """
    with open(os.path.join(last_checkpoint, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)
    saved_per_device_train_batch_size = trainer_state["train_batch_size"]
    saved_max_steps = trainer_state["max_steps"]
    saved_num_train_epochs = trainer_state["num_train_epochs"]

    # [TODO]: save it directly to directly access it
    saved_gradient_accumulation_steps = round(
        (total_rows * saved_num_train_epochs)
        / (saved_max_steps * saved_per_device_train_batch_size)
    )
    start_index = (
        int(last_checkpoint.split("-")[-1])
        * saved_per_device_train_batch_size
        * saved_gradient_accumulation_steps
    )
    return start_index


def tokenize(prompt, tokenizer, cutoff_len: int = None):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
        return_tensors="pt",
    )
    result["input_ids"] = result["input_ids"].flatten()
    result["attention_mask"] = result["attention_mask"].flatten()

    result["labels"] = result["input_ids"].clone()  # Clone input_ids for labels
    return result


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len: int = None):
    if cutoff_len is None:
        tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer)
    else:
        tokenized_full_prompt = tokenize(
            data_point["prompt"], tokenizer=tokenizer, cutoff_len=cutoff_len
        )

        user_prompt = data_point["prompt"].split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[0]
        tokenized_user_prompt = tokenizer(
            # prompt_template.format(data_point["instruction"], data_point["input"]),
            user_prompt,
            max_length=cutoff_len,
            truncation=True,
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        labels_prefix = torch.full((user_prompt_len,), -100)
        tokenized_full_prompt["labels"] = torch.cat(
            (
                labels_prefix,
                torch.tensor(tokenized_full_prompt["labels"][user_prompt_len:]),
            )
        )

    return tokenized_full_prompt


def reorder_dataset(dataset, start_index):
    # Split the dataset into two parts: before and after the start index
    dataset_part1 = dataset.select(range(start_index, len(dataset)))
    dataset_part2 = dataset.select(range(start_index))

    # Concatenate the two parts to get the reordered dataset
    reordered_dataset = concatenate_datasets([dataset_part1, dataset_part2])
    return reordered_dataset


# prints appropriately on console but not to file (.out) trying logger
class PrintExampleCallback(TrainerCallback):
    def __init__(self):
        self.example_iter = None  # Iterator for examples

    def on_train_begin(self, args, state, control, **kwargs):
        self.example_iter = iter(kwargs["train_dataloader"])

    def on_save(self, args, state, control, model, tokenizer, **kwargs):
        # Get the next example from the iterator
        try:
            example = next(self.example_iter)
        except StopIteration:
            # Reinitialize the iterator if it's exhausted and get the next example
            self.example_iter = iter(kwargs["train_dataloader"])
            example = next(self.example_iter)

        logg(
            f"Step {state.global_step}: {tokenizer.decode(example['input_ids'][0], skip_special_tokens=True)}"
        )
