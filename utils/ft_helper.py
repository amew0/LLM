# revise it!!
def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,  # Use the cutoff_len variable you've defined
        padding="max_length",  # Ensure padding is done to the max_length
        return_tensors="pt",  # Return PyTorch tensors
    )
    result["input_ids"] = result["input_ids"].flatten()
    result["attention_mask"] = result["attention_mask"].flatten()

    if add_eos_token and result["input_ids"].shape[0] < cutoff_len:
        # Append eos_token_id to each sequence in the batch
        result["input_ids"][-1] = tokenizer.eos_token_id
        result["attention_mask"][-1] = 1

    result["labels"] = result["input_ids"].clone()  # Clone input_ids for labels
    return result


def generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len):
    tokenized_full_prompt = tokenize(
        data_point["prompt"], tokenizer=tokenizer, cutoff_len=cutoff_len
    )
    return tokenized_full_prompt
