import os
import argparse
import json
import asyncio
import httpx
import time
import aiohttp
import re
from tqdm import tqdm
import logging

from dotenv import load_dotenv

BATCH_SIZE = 30
MODEL = "openai/gpt-4o-2024-11-20"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS = 8192

load_dotenv(dotenv_path="../.env")
API_KEY = os.environ.get("OPENAI_API_KEY")

os.makedirs("./logs", exist_ok=True)

logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppresses INFO logs from httpx

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename='./logs/run.log'
)
logger = logging.getLogger(__name__)

def load_relations(dataset):
    data = json.load(open(f"./relations/{dataset}.json"))
    relations = data["Relations"]
    triples = data["Examples"]
    return list(relations), list(triples)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def create_prompts(relations, triples):
    prompts = []
    batches = list(chunk_list(relations, BATCH_SIZE))
    trip_batches = list(chunk_list(triples, BATCH_SIZE))
    for batch_number, (batch, trip_batch) in enumerate(zip(batches, trip_batches), start=1):
        prompt = return_prompt(batch, trip_batch)
        prompts.append((batch_number, prompt, len(batch)))
    return prompts

def return_prompt(batch_relations, batch_triples):
    relations_str = ""
    for indx, relation in enumerate(batch_relations):
        relations_str += f'relation_name: "{relation}" ; example: "({batch_triples[indx][0]}, {batch_triples[indx][1]}, {batch_triples[indx][2]})"\n'
    # relations_str = "\n".join(batch_relations)
    prompt = f"""You will be provided with a list of relation names, each accompanied by exactly one example triple from a knowledge graph. Follow the instructions below carefully, strictly adhering to the output formats specified.

Step 1: Convert Relation Names to Human-Readable Form

Clean each provided relation name, converting it into plaintext, human-readable form.

Output Format (JSON Dictionary):

{{
"original_relation_name1": "Clean Human-Readable Form",
"original_relation_name2": "Clean Human-Readable Form",
...
}}

Step 2: Generate Short Descriptions

For each provided relation, generate a concise description (3-4 words) that clearly captures its semantic meaning based on the given example triple as context. Also, for each relation, generate a description of its supposed inverse relation. These descriptions will be converted into embeddings using jinaai/jina-embeddings-v3 to uniquely identify relations and to measure semantic similarities. So, avoid using common or generic words excessively, and do NOT reuse other relation names, to prevent false semantic similarities. Follow the rules below,

Be Concise and Precise: Use as few words as possible while clearly conveying the core meaning. Avoid filler words, unnecessary adjectives, and overly generic language.

Emphasize Key Semantics: Focus on the distinctive action or relationship the relation name implies. Ensure that the description highlights the unique aspects that differentiate it from similar relations.

Handle Negation Carefully: If the relation involves negation (e.g., "is not part of"), state the negation explicitly and unambiguously. Ensure that the description for a negated relation is clearly distinguishable from its affirmative counterpart.

Avoid Common Stopwords as Filler: Do not use common stopwords or phrases that add little semantic content. Every word should contribute meaning. Do not use repetitive words to avoid creating false semantic similarities.

Take care of symmetry: Ensure that for relations that are symmetric, the description does not change for its inverse relation.

Output Format (JSON Dictionary):

{{
"original_relation_name1": ["concise description", "concise inverse relation description"],
"original_relation_name2": ["concise description", "concise inverse relation description"],
...
}}

IMPORTANT:

Provide exactly two separate JSON objects in your response, corresponding to each step, strictly in the order presented above. Do not include additional explanations or metadata beyond the specified JSON objects.If you are about to exceed token limits, summarize or shorten the descriptions, but always provide both JSON objects and IT SHOULD CONTAIN ALL RELATIONS AS KEYS.

List of Relations:
{relations_str}
"""
    return prompt

async def async_query_openrouter(session, model, prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt[1]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        # "include_reasoning": True,
        # "top_p": 1,
        # "top_k": 1
    }
    
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Querying OpenRouter with prompt {prompt[0]} of size {prompt[2]}")
            response = await client.post(API_URL, headers=headers, json=data, timeout=2000.0)
            response.raise_for_status()  # Raise an error for 4xx/5xx responses
            try:
                resp_json = response.json()
                logger.info(f"Received response for {prompt[0]}")
            except json.decoder.JSONDecodeError:
                logger.error(f"Failed to decode JSON. Response text: {response.text}")
                return {}
            
            message = resp_json.get("choices", [{}])[0].get("message", {})
            model_ans = message.get("content", "")
            reasoning = message.get("reasoning", "")
            finish_reason = resp_json.get("choices", [{}])[0].get("finish_reason", "")
            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            return {
                "response": model_ans,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "reasoning": reasoning,
                "completion_tokens": completion_tokens
            }
                
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            print(f"Request failed: {e}")
        return None
    

async def batch_call(model, prompts, batch_size=10):
    results = []
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            tasks = [async_query_openrouter(session, model, prompt) for prompt in batch_prompts]
            batch_results = await asyncio.gather(*tasks)
            
            for j, info in enumerate(batch_results):
                if len(info) > 0:
                    context = {"model": model, "prompt": batch_prompts[j], "idx": i+j}
                    context.update(info)
                    results.append(context)
            time.sleep(1)  # Avoid rate limits
    return results

def extract_all_jsons(response_text):
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    json_objects = []
    for match in matches:
        try:
            parsed = json.loads(match)
            json_objects.append(parsed)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue
    return json_objects

def save_intermediate_results(model, results, dataset):
    
    logger.info(f"Saving intermediate results to ./logs/{model}_{dataset}.json")
    with open(f"./logs/{model}_{dataset}.json", "w") as f:
        json.dump(results, f, indent=4)

def merge_results_old(results):
    symm_relations = []
    cleaned_relations = {}
    relation_descriptions = {}
    
    for result in results:
        response = result["response"]
        print(response)
        jsons = extract_all_jsons(response)
        symm_relations.extend(jsons[0])
        cleaned_relations.update(jsons[1])
        relation_descriptions.update(jsons[2])
    res = {"symmetric_relations": symm_relations, "cleaned_relations": cleaned_relations, "relation_descriptions": relation_descriptions}
    # logger.info(f"Merged results:\n{res}")
    print(f"All keys in cleaned_relations: {set(cleaned_relations.keys()) == set(relations)}")
    print(f"All keys in relation_descriptions: {set(relation_descriptions.keys()) == set(relations)}")
    return res

def merge_results(results):
    cleaned_relations = {}
    relation_descriptions = {}
    
    for result in results:
        response = result["response"]
        print(response)
        jsons = extract_all_jsons(response)
        cleaned_relations.update(jsons[0])
        relation_descriptions.update(jsons[1])
    res = {"cleaned_relations": cleaned_relations, "relation_descriptions": relation_descriptions}
    # logger.info(f"Merged results:\n{res}")
    print(f"All keys in cleaned_relations: {set(cleaned_relations.keys()) == set(relations)}")
    print(f"All keys in relation_descriptions: {set(relation_descriptions.keys()) == set(relations)}")
    return res

def save_results(model, results, dataset):
    logger.info(f"Saving results to ./descriptions/{model}/{dataset}.json")
    os.makedirs(f"./descriptions/{model}/", exist_ok=True)
    with open(f"./descriptions/{model}/{dataset}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process relation names from a dataset and query OpenRouter.")
    parser.add_argument("--dataset", required=True, help="Name of the dataset to process.")
    args = parser.parse_args()

    relations, triples = load_relations(args.dataset)
    prompts = create_prompts(relations, triples)

    logger.info(f"Querying {MODEL}")
    results = asyncio.run(batch_call(MODEL, prompts, batch_size=10))
    save_intermediate_results(MODEL.split("/")[1], results, args.dataset)
    results = merge_results(results)
    save_results(MODEL.split("/")[1], results, args.dataset)
    logger.info("Done!")