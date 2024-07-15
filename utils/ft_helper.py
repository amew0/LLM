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


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len: int = None):
    if cutoff_len is None:
        tokenized_full_prompt = tokenize(data_point["prompt"], tokenizer=tokenizer)
    else:
        tokenized_full_prompt = tokenize(
            data_point["prompt"], tokenizer=tokenizer, cutoff_len=cutoff_len
        )
    return tokenized_full_prompt


def reorder_dataset(dataset, start_index):
    # Split the dataset into two parts: before and after the start index
    dataset_part1 = dataset.select(range(start_index, len(dataset)))
    dataset_part2 = dataset.select(range(start_index))

    # Concatenate the two parts to get the reordered dataset
    reordered_dataset = concatenate_datasets([dataset_part1, dataset_part2])
    return reordered_dataset

# prints appropriately on console but not to file (.out)
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


class SFTTrainerNoShuffle(SFTTrainer):
    def _get_train_sampler(self):
        # Using SequentialSampler to prevent shuffling
        return SequentialSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.

        Will use no sampler if `train_dataset` does not implement `__len__`, a sequential sampler otherwise.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
