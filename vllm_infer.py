from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m", download_dir="/dpc/kunf0097/l3-8b/model",
        #   max_seq_length=296
          )
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(max_tokens=100, temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")