import os
import re
import json
import argparse

from openai import OpenAI
from dotenv import load_dotenv
from logger import init_logger, log_chat_response

def load_relations(dataset):
    relations = json.load(open(f"./relations/{dataset}.json"))
    return list(relations)

parser = argparse.ArgumentParser(description="Process relation names from a dataset and query OpenRouter.")
parser.add_argument("--dataset", required=True, help="Name of the dataset to process.")
args = parser.parse_args()

relations = load_relations(args.dataset)

load_dotenv(dotenv_path="../.env")
api_key = os.environ.get("OPENAI_API_KEY")

# Configure OpenRouter via the OpenAI package
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

BATCH_SIZE = 50

def chunk_list(lst, n):
    """Yield successive n-sized chunks from the list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_prompt(batch_relations, batch_number):
    """Create the prompt for a specific batch with a batch identifier."""
    relations_str = "\n".join(batch_relations)
    prompt = f"""You will be provided below a list containing the relation names in a knowledge graph.

First, I want you to identify and list relations which are symmetric. If A is connected to B using R then B is also connected to A using R. For eg: "is connected to", "is related to", "is friend of", etc. Output as a list in JSON format. If there are no symmetric relations, then return an empty list.

Second, I want you to clean the relation name and convert it into plaintext human readable form. Output as a JSON dictionary format where the key is the original relation name provided.

Finally, I want you to describe each of the relations provided below with around 3-4 words which best represents the relation. Whatever description you generate will be converted to an embedding using jinaai/jina-embeddings-v3, which will then be used to define the relations uniquely. They will also be used to compare similarities between different relations - be sure not to use other relation names and too many common words in the descriptions, both of which can create false semantic similarities between relation descriptions. Output as a JSON dictionary format where the key is the original relation name provided.

STRICTLY FOLLOW THE OUTPUT FORMAT AS MENTIONED ABOVE FOR EACH STEP. Output 3 separate JSON objects in the order mentioned above.

List of relations (Batch {batch_number}):
{relations_str}
"""
    return prompt

def extract_all_jsons(response_text):
    """
    Extracts all JSON blocks from the given response text.
    
    Expected format:
    ... ```json
    [JSON block]
    ``` ...
    
    Returns:
      A list of parsed JSON objects in the order they appear.
    """
    # Regex pattern to capture content between ```json and ```
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    json_objects = []
    for match in matches:
        try:
            parsed = json.loads(match)
            json_objects.append(parsed)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            # Optionally, you can raise an error or continue with next block
            continue
    return json_objects


def call_openrouter(prompt):
    """Call OpenRouter using the OpenAI package."""
    try:
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
        return generated_text
    except Exception as e:
        raise Exception(f"OpenRouter API call failed: {e}")

def main():
    logger = init_logger()
    symm_relations = []
    cleaned_relations = {}
    relation_descriptions = {}
    batches = list(chunk_list(relations, BATCH_SIZE))
    
    for batch_number, batch in enumerate(batches, start=1):
        prompt = create_prompt(batch, batch_number)
        log_chat_response(logger, f"Processing batch {batch_number}/{len(batches)} with {len(batch)} relations.")
        log_chat_response(logger, f"Prompt: {prompt}")
        
        try:
            generated_text = call_openrouter(prompt)
            log_chat_response(logger, f"Response: {generated_text}")
            
            jsons = extract_all_jsons(generated_text)
            symm_relations.extend(jsons[0])
            cleaned_relations.update(jsons[1])
            relation_descriptions.update(jsons[2])
        except Exception as e:
            error_msg = f"Error in batch {batch_number}: {e}"
            log_chat_response(logger, error_msg)
            print(error_msg)
    
    combined_results = {"symmetric_relations": symm_relations, "cleaned_relations": cleaned_relations, "relation_descriptions": relation_descriptions}
    with open(f"./descriptions/{args.dataset}.json", "w") as f:
        json.dump(combined_results, f, indent=2)
    combined_results_str = json.dumps(combined_results, indent=2)
    log_chat_response(logger, f"Combined results: {combined_results_str}")
    print("Combined results:")
    print(combined_results_str)

if __name__ == "__main__":
    main()
