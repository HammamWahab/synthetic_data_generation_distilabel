from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps.tasks import TextGeneration
import os 
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

repo_name = "hammamwahab/synthetic_solution_architecture_qwen"

with Pipeline() as pipeline:
    load_dataset = LoadDataFromHub(
        repo_id="VishaalY/solutions-architect-hf-dataset",
        num_examples=3240
    )

    rewrite_problem =  TextGeneration(
        llm=InferenceEndpointsLLM(
            tokenizer_id="Qwen/Qwen2.5-72B-Instruct",
            generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
            # api_key=hf_token,
            base_url=os.getenv("BASE_URL")
        ),
        template="Simplify and rewrite the following problem for AWS with Python. Include only the problem description and nothing else:\n{{ Description }}",
        columns=["Description"],
        output_mappings={"generation": "problem"}
    )

    text_generation = TextGeneration(
        llm=InferenceEndpointsLLM(
            tokenizer_id="Qwen/Qwen2.5-Coder-32B-Instruct",
             generation_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1024,
            },
            # api_key=hf_token,
            base_url=os.getenv("BASE_URL")
        ),
        template="Write a valid and efficient solution in Python for the following problem:\n{{ problem }}",
        columns=["problem"],
        output_mappings={"generation": "response"}
    )
    keep_cols = KeepColumns(columns=['problem', 'response', 'Description', 'Link'])

    load_dataset >> rewrite_problem >> text_generation >> keep_cols

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    distiset.push_to_hub(
        repo_id=repo_name,
        generate_card=True,
        include_script=True,
    )