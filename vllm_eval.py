from dotenv import load_dotenv
from openai import OpenAI
import fire
from utils.eval_helper import log2json
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

logg = lambda x: print(f"------------------------ {x} ---------------------------")

load_dotenv()


def get_prompt(template, example):
    template[1]["content"] = template[1]["content"].format(
        example["question"], example["choices"]
    )
    return template


def get_response(text: str):
    # for now only strip it
    return text.strip()


def main(
    base_url: str = "",
    candidate_name="meta-llama/Meta-Llama-3-8B-Instruct",
    eval_data_path="data/eval_medical_2k.json",
):
    """
    Using OpenAI entrypoints

    Args:
    base_url (str): URL to use for inference. Default is "".
    """
    run_id = datetime.now().strftime("%y%m%d%H%M%S")

    client = OpenAI(base_url=base_url)
    generation_config = {"temperature": 0.7, "max_tokens": 150}
    evaluator_template = [
        {
            "role": "system",
            "content": "You are going to be evaluated on a medical datatet. Choices will be given and your task is to select the best answer. Since exact matches is going to be used as an evaluation make sure YOU GENERATE EXACTLY THE CORRECT ANSWER and NO FURTHER EXPLANATION OR INTRODUCTION.",
        },
        {"role": "user", "content": "### Question: {}\n\nChoices: {}\n\nAnswer: "},
    ]

    # data = load_dataset("json", data_files=eval_data_path)
    # eval_dataset = data["train"]
    data = load_dataset("cais/mmlu", "clinical_knowledge")
    eval_dataset = data["dev"]

    records = []
    for i, example in tqdm(enumerate(eval_dataset)):
        evaluator_prompt = get_prompt(evaluator_template, example)

        completion = client.chat.completions.create(
            model=candidate_name,
            messages=evaluator_prompt,
            stop=["\n### Human"],
            **generation_config,
        )

        res = completion.choices[0].message.content
        res = get_response(res)

        record = {
            "question": example["question"],
            "expected": example["choices"][example["answer"]],
            "generated": res,
        }
        records.append(record)

        log2json(records, f"attempts_{candidate_name.split('/')[1]}_{run_id}.json")


if __name__ == "__main__":
    fire.Fire(main)
