import datasets
from datasets import concatenate_datasets
from transformers import TrainerCallback
from transformers.trainer_utils import seed_worker
import torch
from torch.utils.data import DataLoader, SequentialSampler
from trl import SFTTrainer

from utils.eval_helper import logg


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


def generate_and_tokenize_prompt(
    data_point, tokenizer, cutoff_len: int = None, prompt_template: str = None
):
    train_on_input = False
    if cutoff_len is None:
        tokenized_full_prompt = tokenize(
            prompt_template.format(
                data_point["instruction"], data_point["input"], data_point["output"]
            ),
            tokenizer=tokenizer,
        )
    else:
        tokenized_full_prompt = tokenize(
            prompt_template.format(
                data_point["instruction"], data_point["input"], data_point["output"]
            ),
            tokenizer=tokenizer,
            cutoff_len=cutoff_len,
        )
        if not train_on_input:
            prompt_template = prompt_template.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[0]
            tokenized_user_prompt = tokenize(
                prompt_template.format(data_point["instruction"], data_point["input"]),
                tokenizer=tokenizer,
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
