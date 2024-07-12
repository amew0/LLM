from dotenv import load_dotenv
from openai import OpenAI
import fire
from utils.eval_helper import log2json, extract_digit
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm


logg = lambda x: print(f"------------------------ {x} ---------------------------")

load_dotenv()


def get_prompt(template, example):
    template[1]["content"] = template[1]["content"].format(example["input"], example["output"])
    return template


def get_qnas(qnas):
    try:
        start_index = qnas.find("### 1")
        qna_section = qnas[start_index:].strip()

        # Skip the empty string before the first ###
        qna_blocks = qna_section.split("### ")[1:]

        qnas_formatted = []
        for block in qna_blocks:
            qna = block.strip().split("\n")[1:]
            q = qna[0]
            m = qna[1]
            a = qna[2]

            qnas_formatted.append({"Q": q, "M": m, "A": a, "QnA": block})
            if len(qnas_formatted) > 5:
                break
        return qnas_formatted
    except Exception as e:
        print(qnas)
        print(e)
        return None


def main(
    base_url: str = "",
    evaluator_name="meta-llama/Meta-Llama-3-8B-Instruct",
    eval_data_path="data/eval_medical_2k.json",
):
    """
    Using OpenAI entrypoints

    Args:
    base_url (str): URL to use for inference. Default is "".
    evaluator_name (str): Model name for the evaluator. Default is 'meta-llama/Meta-Llama-3-8B-Instruct'.
    eval_data_path (str): Path to the evaluation data. Default is './data/1/eval_medical_2k.json'.
    """

    run_id = datetime.now().strftime("%y%m%d%H%M%S")

    client = OpenAI(base_url=base_url)
    generation_config = {
        "temperature": 0.7,
        "max_tokens": 300 ,
    }
    evaluator_template = [
        {
            "role": "system",
            "content": "You are a Multiple Choice Question generator with its Answer Letter (A, B, C, D) from a provided CONTEXT. YOU DONT PRINT OUT ANYTHING ELSE EXCEPT THE MCQs. The context is a dialog between a patient (or someone on their behalf) and a doctor over text. I need 5 questions and for each question there should be 4 choices. The questions generated should address different aspects of the problem with multiple choices that's not easy to guess. \n1. Don't ask questions where their answer is already simply there in the patient’s history. \n2. DONT ASK ANY NUMERICAL questions since they are easily spottable from the patient’s history (hence don’t qualify as challenging questions).\nThe questions should require domain knowledge, critical thinking, and should not be easily answerable by looking at the question. Refer the doctor's answer, the diagnosis, the patient's symptoms, and other factors to come up with the questions. On your question you are allowed to end it or start with ONLY `based on the given context` and nothing else like 'based on the doctor's ...'! For the answer I want only the label without parenthesis or anything else just the letter! \nSTRICTLY FOLLOW THIS FORMAT:\n### 1\nQ: [Question]\nM: A) ..., B) ..., C) ..., D) ...\nA: [A or B or C or D]\n### 2\n",
        },
        {
            "role": "user",
            "content": "Here is some instruction: 1. the format stated is very important\n2. No mentioning of the doctor in your question.\n3. And more important I want you to generate only the MCQ and dont say anything else!\n\n### Patient (or behalf): {}\n\n### Doctor (over text): {}",
        },
        # {
        #     "role": "assistant",
        #     "content": "To clarfiy, you want 5 Non numeric Q&A with 4 choices based on the provided cotext and strictly follow the format, right?",
        # },
        # {
        #     "role": "user",
        #     "content": "exactly, the format is very important, make sure the Answer is in the choices, and no mentioning of the doctor in your question. And more important I want you to generate for me only the MCQ dont say anything else",
        # },
    ]

    data = load_dataset("json", data_files=eval_data_path)
    eval_dataset = data["train"]

    records = []
    for i, example in tqdm(enumerate(eval_dataset)):
        evaluator_prompt = get_prompt(evaluator_template, example)

        completion = client.chat.completions.create(
            model=evaluator_name,
            messages=evaluator_prompt,
            stop=["\n### Human"],
            **generation_config,
        )

        qnas = completion.choices[0].message.content
        # questions = get_qnas(qnas)

        # if questions is None:
        #     continue

        record = {"context": example["input"], "questions": qnas}
        records.append(record)

        log2json(records, f"questions_{run_id}.json")


if __name__ == "__main__":
    fire.Fire(main)
