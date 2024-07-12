qnas = " \n\n### 1\nQ: What is the patient's liver condition based on the provided lab results?\nM: cirrhosis, chronic hepatitis, liver damage, liver failure\nA: chronic hepatitis\n\n### 2\nQ: What is the patient's serum albumin level?\nM: 3.4, 7.54, 11.3, 15.6\nA: 3.4\n\n### 3\nQ: Which of the following is a recommendation for the patient's treatment?\nM: avoid fatty diet, take beta blocker, take more sugar cane juice, consult gastroenterologist\nA: consult gastroenterologist\n\n### 4\nQ: What is the patient's bilirubin level?\nM: 2.38, 4.88, 17.16, 25.8\nA: 17.16\n\n### 5\nQ: What is the patient's creatinine level?\nM: 2.5, 4.88, 7.5, 10.5\nA: 4.88\n"


def get_qnas(qnas):
    try:
        start_index = qnas.find("### 1")

        qna_section = qnas[start_index:].strip()

        # Skip the empty string before the first ###
        qna_blocks = qna_section.split("### ")[1:]

        l = str(3)
        l.find
        qnas_formatted = []
        for block in qna_blocks:
            qna = block.strip()
            q = qna[qna.find("Q: ") : qna.find("A: ")].strip()
            a = qna[qna.find("A: ") : qna.find("\n", qna.find("A: "))].strip()

            qnas_formatted.append({"Q": q, "A": a, "QnA": qna})
    except Exception as e:
        print(qnas)
        print(e)
        return None


get_qnas(qnas)
