# LLM Evaluation Framework

A comprehensive framework for evaluating Large Language Models (LLMs) using multiple evaluation strategies, including exact match testing for multiple-choice questions and LLM-as-a-judge approaches.

## 🎯 Features

- Multiple evaluation strategies:
  - Exact Match evaluation for multiple-choice questions
  - LLM-as-a-judge approach for open-ended responses
  - Support for multiple evaluators per response
- Support for various LLM backends:
  - Hugging Face models
  - OpenAI-compatible APIs
  - vLLM inference server
- Flexible dataset handling with support for custom and standard benchmarks
- Wandb integration for experiment tracking
- Comprehensive logging and result analysis

## 🚀 Quick Start

### Prerequisites
bash
pip install -r requirements.txt

### Environment Setup

Create a `.env` file with your credentials:
```env
HF_TOKEN_WRITE=your_huggingface_token
OPENAI_API_KEY=your_openai_key  # if using OpenAI
```

### Running Evaluations

1. **LLM-as-a-Judge Evaluation**
bash:README.md
python eval_pipeline.py \
--candidate_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--evaluator_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--eval_data_path="./data/eval_medical_2k.json" \
--evals_per_example=2


2. **Multiple Choice Generation**

bash
python eval_generate.py \
--base_url="your_vllm_endpoint" \
--evaluator_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--eval_data_path="data/eval_medical_2k.json"


bash
python vllm_eval.py \
--base_url="your_vllm_endpoint" \
--candidate_name="meta-llama/Meta-Llama-3-8B-Instruct" \
--eval_data_path="data/eval_medical_2k.json"

## 📊 Evaluation Methods

### 1. LLM-as-a-Judge
- Uses one LLM to evaluate the responses of another
- Supports multiple evaluations per response for reliability
- Calculates Pearson correlation coefficient between evaluator scores
- Provides running averages and detailed scoring metrics

### 2. Multiple Choice Evaluation
- Exact match comparison for multiple-choice questions
- Supports standard benchmarking datasets (e.g., MMLU)
- Automated question generation from context
- Structured output format for easy analysis

## 📁 Project Structure
.
├── eval_pipeline.py # Main LLM-as-a-judge evaluation pipeline
├── eval_generate.py # MCQ generation from context
├── vllm_eval.py # Multiple choice evaluation
├── utils/
│ └── eval_helper.py # Shared utilities and helper functions
└── template.yaml # Prompt templates configuration


## 📈 Output Format

Results are saved in JSON format with detailed metrics:
- Individual response scores
- Running averages
- Inter-evaluator correlation
- Detailed evaluation metadata

## 🔧 Configuration

### template.yaml
Configure prompt templates and generation parameters:

```yaml

model_name:
candidate_prompt: "..."
evaluator_prompt: "..."
generation_config:
max_new_tokens: 256
temperature: 0.6
top_p: 0.9
```
