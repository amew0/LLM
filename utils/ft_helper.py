import os
import json
from datasets import concatenate_datasets
from transformers import TrainerCallback
import inspect


def logg(x):
    print(f"------------------------ {x} ---------------------------")


def inspectt(frame):
    logg("")
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(f"\t{arg}: {values[arg]}")
    logg("")


def get_start_index(last_checkpoint, total_rows) -> int:
    """
    Get the start index to resume training from the last checkpoint.

    Args:
        last_checkpoint: str
        total_rows: int
    Returns:
        start_index: int
    """
    # a simpler and easier eq.n
    # start_index = (total_rows * epoch) % total_rows
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
    ) % total_rows
    return start_index


# [ARCHIVE]
# def tokenize(prompt, tokenizer, cutoff_len: int = None, add_eos_token=True):
#     result = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=cutoff_len,
#         padding=False,
#         return_tensors=None,
#     )

#     # if truncated the last token is not eostoken so we need to add it
#     if (
#         result["input_ids"][-1] != tokenizer.eos_token_id
#         and len(result["input_ids"]) < cutoff_len
#         and add_eos_token
#     ):
#         # append or replace?
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)
#     result["labels"] = result["input_ids"].copy()
#     return result


# def generate_and_tokenize_prompt(
#     data_point, tokenizer, cutoff_len: int = None, train_on_inputs=False
# ):
#     full_prompt = data_point["prompt"]

#     tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len)

#     if not train_on_inputs:

#         # user_prompt = prompt_template.format(data_point["instruction"], data_point["input"])
#         user_prompt = full_prompt.split("<|start_header_id|>assistant<|end_header_id|>")[0]

#         tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len)

#         user_prompt_len = len(tokenized_user_prompt["input_ids"])
#         labels_prefix = [-100] * user_prompt_len
#         tokenized_full_prompt["labels"] = (
#             labels_prefix + tokenized_full_prompt["labels"][user_prompt_len:]
#         )

#     return tokenized_full_prompt


# [WIP]
def tokenize(prompt, tokenizer, cutoff_len: int = None):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )

    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(
    example, tokenizer, ft_config, cutoff_len: int = None, train_on_inputs=False
):
    user_prompt = ft_config["prompt"].format(example["instruction"], example["input"])
    response = ft_config["response"].format(example["output"])
    full_prompt = user_prompt + response

    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len)

    if not train_on_inputs:
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len)

        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        labels_prefix = [-100] * user_prompt_len
        tokenized_full_prompt["labels"] = (
            labels_prefix + tokenized_full_prompt["labels"][user_prompt_len:]
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


# [Just in case]
# def format_prompt(batch, ft_config):
#     # not the best approach
#     prompts = []
#     for i in range(len(batch['instruction'])):
#         example = {key: value[i] for key, value in batch.items()}

#         user_prompt = ft_config["prompt"].format(example["instruction"], example["input"])
#         response = ft_config["response"].format(example["output"])
#         full_prompt = user_prompt + response
#         prompts.append(full_prompt)

#     return prompts

# # def collate_fn(batch, ft_config, cutoff_len):
# response_template = "\n### Assistant:"
# from trl import DataCollatorForCompletionOnlyLM
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
