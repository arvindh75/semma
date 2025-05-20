import os
import argparse
import json
import re

from dotenv import load_dotenv
from openai import OpenAI

MODEL = "deepseek/deepseek-chat"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 8192

load_dotenv(dotenv_path="../.env")
API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

def load_relations(dataset):
    relations = json.load(open(f"./relations/{dataset}.json"))
    return list(relations)

def extract_all_jsons(response_text):
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    json_objects = []
    for match in matches:
        try:
            parsed = json.loads(match)
            json_objects.append(parsed)
        except json.JSONDecodeError as e:
            print(response_text)
            print(f"Error decoding JSON: {e}")
            # Optionally, you can raise an error or continue with next block
            continue
    return json_objects

def call_openrouter(prompt):
    try:
        print(f"Querying OpenRouter")
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        generated_text = response.choices[0].message.content
        print(f"Received response")
        return generated_text
    except Exception as e:
        raise Exception(f"OpenRouter API call failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process relation names from a dataset and query OpenRouter.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset to process.")
    args = parser.parse_args()

    relations = load_relations(args.dataset)

    symm_relations = []
    cleaned_relations = {}
    relation_descriptions = {}

    with open(f"./logs/{MODEL.split('/')[1]}_{args.dataset}.json", "r") as f:
        results = json.load(f)
    
    for result in results:
        response = result["response"]
        if response == "":
            response = call_openrouter(result["prompt"][1])
        jsons = extract_all_jsons(response)
        symm_relations.extend(jsons[0])
        try:
            cleaned_relations.update(jsons[1]["cleaned_relations"])
        except:
            cleaned_relations.update(jsons[1])
        try:
            relation_descriptions.update(jsons[2]["relation_descriptions"])
        except:
            relation_descriptions.update(jsons[2])
    combined_results = {"symmetric_relations": symm_relations, "cleaned_relations": cleaned_relations, "relation_descriptions": relation_descriptions}
    with open(f"./descriptions/{args.dataset}.json", "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"All keys in cleaned_relations: {set(cleaned_relations.keys()) == set(relations)}")
    print(f"All keys in relation_descriptions: {set(relation_descriptions.keys()) == set(relations)}")
