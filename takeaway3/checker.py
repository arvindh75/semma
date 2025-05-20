import os
from collections import defaultdict

# Define file paths relative to the script location or use absolute paths
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
raw_dir = os.path.join(script_dir, "mtdea/WikiTopics-MT4/health/raw")

inf_test_path = os.path.join(raw_dir, "inf_test_new.txt")
inference_graph_path = os.path.join(raw_dir, "inference_graph_new.txt")

# --- Step 1 & 3: Read inference_graph.txt and extract heads/relations ---
inference_heads = set()
inference_relations = set()

print(f"Reading inference graph: {inference_graph_path}")
try:
    with open(inference_graph_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                head, relation, _ = parts
                inference_heads.add(head)
                inference_relations.add(relation)
            # else:
            #     print(f"Skipping malformed line in inference_graph.txt: {line.strip()}")
except FileNotFoundError:
    print(f"Error: File not found - {inference_graph_path}")
    exit(1)
print(f"Found {len(inference_heads)} unique heads and {len(inference_relations)} unique relations in inference graph.")


# --- Step 2 & 4: Read inf_test.txt, store triples, and identify initial candidate heads ---
test_triples_by_head = defaultdict(list)
candidate_heads = set()

print(f"Reading test file: {inf_test_path}")
try:
    with open(inf_test_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                head, relation, tail = parts
                test_triples_by_head[head].append((relation, tail))

                # Check criteria: head seen in inference, relation not seen in inference
                if head in inference_heads and relation not in inference_relations:
                    candidate_heads.add(head)
            else:
                print(relation)
                print(f"Skipping malformed line in inf_test.txt: {line.strip()}")
except FileNotFoundError:
    print(f"Error: File not found - {inf_test_path}")
    exit(1)
print(f"Found {len(candidate_heads)} initial candidate heads satisfying the criteria.")

# --- Step 5 & 6: Filter candidates to keep only those with multiple relations in inf_test.txt ---
multi_relation_count = 0
final_candidates_data = defaultdict(list)

print("Filtering candidates and counting those with multiple relations...")
for head in candidate_heads:
    if head in test_triples_by_head:
        relations_for_head = test_triples_by_head[head]
        if len(relations_for_head) > 1:
            multi_relation_count += 1
            final_candidates_data[head] = relations_for_head # Store if needed later

# --- Step 7: Output the count ---
print(f"\nFound {multi_relation_count} candidate heads that satisfy the criteria AND have multiple relations in inf_test.txt.")

# --- Step 8: Calculate and print unique relations and their tail counts per candidate head ---
print("\nDetails for candidate heads (sorted by number of unique relations):")
head_details = [] # To store (head, num_unique_relations, {relation: num_unique_tails})

for head, triples in final_candidates_data.items():
    relations_to_tails = defaultdict(set)
    for rel, tail in triples:
        relations_to_tails[rel].add(tail)

    num_unique_relations = len(relations_to_tails)
    relation_tail_counts = {rel: len(tails) for rel, tails in relations_to_tails.items()}

    head_details.append((head, num_unique_relations, relation_tail_counts))

# Sort by number of unique relations descending
head_details.sort(key=lambda item: item[1], reverse=True)

# Print the results
for head, num_rels, rel_tail_counts in head_details:
    print(f"Head: {head} ({num_rels} unique relations)")
    # Sort relations alphabetically for consistent output
    sorted_relations = sorted(rel_tail_counts.items(), key=lambda item: item[0])
    for rel, tail_count in sorted_relations:
        print(f"  \tRelation: {rel}, Unique Tails: {tail_count}")

# Optional: Print the final candidate heads and their triples if needed
# print("\nCandidate heads and their relations/tails:")

# check if any relation of test is present in inference
# for head, triples in final_candidates_data.items():
#     for rel, tail in triples:
#         if rel in inference_relations:
#             print(f"Relation {rel} found in both test and inference.")
#         else:
#             print(f"Relation {rel} found only in test.")