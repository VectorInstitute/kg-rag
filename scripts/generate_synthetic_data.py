"""Generate synthetic data with numerical/objective answers based on the input CSV."""

import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def generate_synthetic_data(input_csv_path, output_csv_path):
    """
    Generate a synthetic dataset with numerical/objective answers based on the input CSV.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to the output CSV file.
    """
    df = pd.read_csv(input_csv_path)

    llm = ChatOpenAI(temperature=0, model="gpt-4o")  # type: ignore

    prompt_template = """
    Given the following question and answer, generate a new question that requires a numerical or objective answer, and provide the numerical or objective answer.
    The new question should be at one of three levels:
    - level 1: single chunk.
    - level 2: multi-doc/multi-chunk
    - level 3: multi-step reasoning

    Here is a good example of the specificity we are looking for when describing the format of the answers:

    Level 1
    Question: What was the actual enrollment count of the clinical trial on H. pylori in acne vulgaris
    patients from Jan-May 2018 as listed on the NIH website?
    Ground truth: 90

    Level 2
    Question: If this whole pint is made up of ice cream, how many percent above
    or below the US federal standards for butterfat content is it when using the
    standards as reported by Wikipedia in 2020? Answer as + or - a number rounded
    to one decimal place.
    Ground truth: +4.6

    Level 3
    Question: In NASA’s Astronomy Picture of the Day on 2006 January 21, two astronauts are visible,
    with one appearing much smaller than the other. As of August 2023, out of the astronauts in the
    NASA Astronaut Group that the smaller astronaut was a member of, which one spent the least time
    in space, and how many minutes did he spend in space, rounded to the nearest minute? Exclude any
    astronauts who did not spend any time in space. Give the last name of the astronaut, separated from
    the number of minutes by a semicolon. Use commas as thousands separators in the number of minutes.
    Ground truth: White; 5876

    Remember to specify the format of the answer in the new question. Make sure to specify units if necessary. The answer should only be a numerical value that can be easily verified and aligned with the units specified in the question.

    Also specify the level as an integer in a separate column.

    Question: {question}
    Answer: {answer}

    New Question:
    New Answer:
    Level:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = LLMChain(llm=llm, prompt=prompt)

    new_questions = []
    new_answers = []
    levels = []

    for _, row in df.iterrows():
        question = row["Question"]
        answer = row["Answer"]
        output = chain.run(question=question, answer=answer)
        try:
            new_question = (
                output.split("New Question:")[1].split("New Answer:")[0].strip()
            )
            new_answer = output.split("New Answer:")[1].split("Level:")[0].strip()
            level = output.split("Level:")[1].strip()
            new_questions.append(new_question)
            new_answers.append(new_answer)
            levels.append(level)
        except IndexError:
            new_questions.append("")
            new_answers.append("")
            levels.append("")

    df["New Question"] = new_questions
    df["New Answer"] = new_answers
    df["Level"] = levels
    df.to_csv(output_csv_path, index=False)
    print(f"Synthetic data generated and saved to {output_csv_path}")


if __name__ == "__main__":
    input_csv_path = "data/sec-10-q/qna_data.csv"
    output_csv_path = "data/sec-10-q/synthetic_qna_data.csv"
    generate_synthetic_data(input_csv_path, output_csv_path)
