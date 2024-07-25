# to connect the trained lora to the base model
# needs revising was cut from ft.py

# # saving to load later from https://www.youtube.com/watch?v=Pb_RGAl75VE&ab_channel=DataCamp
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     cache_dir=f"{cache_dir}/model",
#     torch_dtype=torch.float16,
#     device_map={"": 0},
#     return_dict=True,
# )
# model = PeftModel.from_pretrained(model, model_save_path)
# model = model.merge_and_unload()  # revise it!!

# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     cache_dir=f"{cache_dir}/tokenizer",
#     padding_side="right",
#     pad_token_id=(0),
#     legacy=False,
# )
# tokenizer.pad_token = tokenizer.eos_token

# # Push to Hugging Face Hub
# tokenizer.push_to_hub(f"{model_name}-v{run_id}", token=HF_TOKEN_WRITE)
# model.push_to_hub(f"{model_name}-v{run_id}", token=HF_TOKEN_WRITE)