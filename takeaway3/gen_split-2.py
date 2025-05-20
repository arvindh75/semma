import os
import random
from collections import defaultdict

# --- Configuration ---
# Define file paths relative to the script location or use absolute paths
script_dir = os.path.dirname(__file__)
raw_dir = os.path.join(script_dir, "mtdea/WikiTopics-MT1/tax/raw")

original_inf_graph_path = os.path.join(raw_dir, "inference_graph.txt")
original_inf_test_path = os.path.join(raw_dir, "inf_test.txt")

new_inf_graph_path = os.path.join(raw_dir, "inference_graph_new.txt")
new_inf_test_path = os.path.join(raw_dir, "inf_test_new.txt")

# Parameters for the split
MIN_RELATIONS_PER_TEST_HEAD = 3  # Minimum number of distinct relations a head must have in test set
TEST_RELATION_FRACTION = 0.2  # Percentage of relation types to include in test set
MIN_HEADS_PER_TEST_RELATION = 1  # Minimum number of different heads per test relation

# User-defined target for test set size
USER_TARGET_MIN_TEST_TRIPLES = 700
USER_TARGET_MAX_TEST_TRIPLES = 1000

# --- Helper Functions ---
def read_triples_and_relations(filepath):
    """Reads triples and unique relations from a file."""
    triples = set()
    relations = set()
    print(f"Reading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, r, t = parts
                    triples.add((h, r, t))
                    relations.add(r)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        exit(1)
    print(f"Found {len(triples)} unique triples and {len(relations)} unique relations.")
    return triples, relations

def write_triples(filepath, triples):
    """Writes triples to a file, one per line."""
    print(f"Writing {len(triples)} triples to {filepath}...")
    with open(filepath, 'w') as f:
        sorted_triples = sorted(list(triples))
        for h, r, t in sorted_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print("Writing complete.")

def analyze_head_connectivity(triples):
    """Maps each head entity to its relations and their counts."""
    head_to_relations = defaultdict(set)
    head_relation_counts = defaultdict(lambda: defaultdict(int))
    
    for h, r, t in triples:
        head_to_relations[h].add(r)
        head_relation_counts[h][r] += 1
    
    # Sort heads by the number of distinct relations they have
    heads_by_relation_count = sorted(
        [(h, len(relations)) for h, relations in head_to_relations.items()],
        key=lambda x: x[1], 
        reverse=True
    )
    
    return head_to_relations, head_relation_counts, heads_by_relation_count

# --- Main Logic ---

# 1. Read original data
all_triples, all_relations = read_triples_and_relations(original_inf_graph_path)
test_triples, _ = read_triples_and_relations(original_inf_test_path)
combined_triples = all_triples.union(test_triples)

print(f"\nOriginal data statistics:")
print(f"Total unique triples: {len(combined_triples)}")
print(f"Total unique relations: {len(all_relations)}")

# 2. Analyze entity connectivity
head_to_relations, head_relation_counts, heads_by_relation_count = analyze_head_connectivity(combined_triples)
print(f"\nFound {len(heads_by_relation_count)} unique heads.")
print(f"Top 5 heads by relation count: {heads_by_relation_count[:5]}")

# 3. Calculate the relation statistics
relation_to_triples = defaultdict(set)
relation_to_heads = defaultdict(set)

for h, r, t in combined_triples:
    relation_to_triples[r].add((h, r, t))
    relation_to_heads[r].add(h)

# Sort relations by number of distinct heads they connect
relations_by_head_count = sorted(
    [(r, len(heads), len(relation_to_triples[r])) for r, heads in relation_to_heads.items()],
    key=lambda x: (x[1], x[2]),  # Sort by head count, then by triple count
    reverse=True
)

# 4. Select relations for the test set
# We want to select relations that have enough distinct heads
num_test_relations = max(1, int(len(all_relations) * TEST_RELATION_FRACTION))
test_relations = set()

for r, head_count, triple_count in relations_by_head_count:
    if head_count >= MIN_HEADS_PER_TEST_RELATION:
        test_relations.add(r)
        if len(test_relations) >= num_test_relations:
            break

print(f"\nSelected {len(test_relations)} relations for the test set.")

# 5. Extract all potential test triples (triples with test relations)
potential_test_triples = {triple for triple in combined_triples if triple[1] in test_relations}
print(f"Found {len(potential_test_triples)} potential test triples.")

# 6. Create initial inference graph (with all triples except potential test triples)
initial_inf_graph = combined_triples - potential_test_triples

# Extract all entities in the inference graph 
inf_entities = set()
for h, _, t in initial_inf_graph:
    inf_entities.add(h)
    inf_entities.add(t)

print(f"\nInitial inference graph contains {len(initial_inf_graph)} triples and {len(inf_entities)} entities.")

# 7. Find all potential test triples where both head and tail exist in the inference graph
valid_test_candidates = {
    (h, r, t) for h, r, t in potential_test_triples 
    if h in inf_entities and t in inf_entities
}

print(f"Found {len(valid_test_candidates)} valid test triple candidates.")

# 8. Group valid test candidates by head
head_to_valid_test_triples = defaultdict(list)
for h, r, t in valid_test_candidates:
    head_to_valid_test_triples[h].append((h, r, t))

# Calculate how many distinct relations each head has in the valid test candidates
head_to_distinct_test_relations = {
    h: len(set(triple[1] for triple in triples))
    for h, triples in head_to_valid_test_triples.items()
}

# Filter heads that have at least MIN_RELATIONS_PER_TEST_HEAD distinct relations
qualifying_heads = [
    h for h, rel_count in head_to_distinct_test_relations.items() 
    if rel_count >= MIN_RELATIONS_PER_TEST_HEAD
]

print(f"Found {len(qualifying_heads)} heads with at least {MIN_RELATIONS_PER_TEST_HEAD} distinct test relations.")

# 9. Build the test set
selected_test_triples = set()
head_to_selected_relations = defaultdict(set)

# First, process heads with enough relations
for head in qualifying_heads:
    # Group triples by relation for this head
    triples_by_relation = defaultdict(list)
    for triple in head_to_valid_test_triples[head]:
        _, r, _ = triple
        triples_by_relation[r].append(triple)
    
    # Select at least MIN_RELATIONS_PER_TEST_HEAD distinct relations
    selected_relations = set()
    
    # Sort relations by number of triples (to prioritize relations with more examples)
    relations_sorted = sorted(
        triples_by_relation.keys(),
        key=lambda r: len(triples_by_relation[r]),
        reverse=True
    )
    
    # Select the top MIN_RELATIONS_PER_TEST_HEAD relations
    for r in relations_sorted[:MIN_RELATIONS_PER_TEST_HEAD]:
        selected_relations.add(r)
        # Add all triples for this relation
        for triple in triples_by_relation[r]:
            selected_test_triples.add(triple)
            head_to_selected_relations[head].add(r)

print(f"Initial test set contains {len(selected_test_triples)} triples.")

# 10. Ensure test set size is within target range
current_test_size = len(selected_test_triples)

if current_test_size > USER_TARGET_MAX_TEST_TRIPLES:
    print(f"Test set size ({current_test_size}) exceeds max target ({USER_TARGET_MAX_TEST_TRIPLES}). Reducing...")
    
    # We need to remove some triples while maintaining the minimum relation requirements
    while len(selected_test_triples) > USER_TARGET_MAX_TEST_TRIPLES:
        # Identify heads with the most relations
        heads_by_relation_count = sorted(
            [(h, len(rels)) for h, rels in head_to_selected_relations.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Start removing from heads with the most relations
        for head, rel_count in heads_by_relation_count:
            if rel_count > MIN_RELATIONS_PER_TEST_HEAD:
                # Find a relation to remove for this head
                head_relations = list(head_to_selected_relations[head])
                rel_to_remove = head_relations[-1]  # Remove the "least important" relation
                
                # Find and remove triples with this head and relation
                triples_to_remove = {
                    (h, r, t) for h, r, t in selected_test_triples
                    if h == head and r == rel_to_remove
                }
                
                if triples_to_remove:
                    selected_test_triples -= triples_to_remove
                    head_to_selected_relations[head].remove(rel_to_remove)
                    print(f"  Removed {len(triples_to_remove)} triples with relation {rel_to_remove} from head {head}")
                    
                    if len(selected_test_triples) <= USER_TARGET_MAX_TEST_TRIPLES:
                        break
            
            if len(selected_test_triples) <= USER_TARGET_MAX_TEST_TRIPLES:
                break
        
        # If we can't reduce further without violating constraints, break
        if len(selected_test_triples) > USER_TARGET_MAX_TEST_TRIPLES:
            # Check if any further reduction is possible
            can_reduce_more = any(
                len(rels) > MIN_RELATIONS_PER_TEST_HEAD 
                for rels in head_to_selected_relations.values()
            )
            
            if not can_reduce_more:
                print("  Cannot reduce test set further without violating constraints.")
                break

elif current_test_size < USER_TARGET_MIN_TEST_TRIPLES:
    print(f"Test set size ({current_test_size}) is below min target ({USER_TARGET_MIN_TEST_TRIPLES}). Attempting to add more...")
    
    # Try to add more triples from valid candidates
    remaining_candidates = valid_test_candidates - selected_test_triples
    
    # Sort remaining candidates by how many relations the head already has
    # (prioritize adding triples to heads with fewer relations)
    head_relation_counts = defaultdict(int)
    for h, r, t in selected_test_triples:
        head_relation_counts[h] += 1
    
    sorted_candidates = sorted(
        remaining_candidates,
        key=lambda x: (head_relation_counts.get(x[0], 0), x[0], x[1])
    )
    
    # Add candidates until we reach the minimum
    for triple in sorted_candidates:
        if len(selected_test_triples) >= USER_TARGET_MIN_TEST_TRIPLES:
            break
            
        selected_test_triples.add(triple)
        h, r, _ = triple
        head_to_selected_relations[h].add(r)
    
    print(f"  Added {len(selected_test_triples) - current_test_size} more triples to test set.")

# 11. Finalize the split
# All relations in the test set should not appear in the inference graph
final_test_set = selected_test_triples
final_inf_graph = combined_triples - final_test_set

# Verify no relation overlap
test_relations = {r for _, r, _ in final_test_set}
inf_relations = {r for _, r, _ in final_inf_graph}
relation_overlap = test_relations.intersection(inf_relations)

if relation_overlap:
    print(f"\nWARNING: {len(relation_overlap)} relations appear in both inference and test sets!")
    print("Fixing relation overlap by moving triples with overlapping relations from inference to test...")
    
    # Move triples with overlapping relations from inference to test
    overlapping_triples = {
        (h, r, t) for h, r, t in final_inf_graph
        if r in relation_overlap
    }
    
    final_inf_graph -= overlapping_triples
    
    # Do NOT add these to the test set, as we want relations to be exclusive to test
    print(f"Removed {len(overlapping_triples)} triples with overlapping relations from inference graph.")
    
    # Re-check
    inf_relations = {r for _, r, _ in final_inf_graph}
    relation_overlap = test_relations.intersection(inf_relations)
    if not relation_overlap:
        print("Successfully fixed relation overlap.")
    else:
        print(f"WARNING: Still have {len(relation_overlap)} overlapping relations.")

# 12. Verify all entities in test set exist in inference graph
test_entities = set()
for h, _, t in final_test_set:
    test_entities.add(h)
    test_entities.add(t)

inf_entities = set()
for h, _, t in final_inf_graph:
    inf_entities.add(h)
    inf_entities.add(t)

missing_entities = test_entities - inf_entities
if missing_entities:
    print(f"\nWARNING: {len(missing_entities)} entities in test set are missing from inference graph!")
    print("Fixing by moving triples with missing entities from test to inference...")
    
    # Identify triples with missing entities
    triples_with_missing_entities = {
        (h, r, t) for h, r, t in final_test_set
        if h in missing_entities or t in missing_entities
    }
    
    # Move these triples to inference graph
    final_test_set -= triples_with_missing_entities
    final_inf_graph |= triples_with_missing_entities
    
    print(f"Moved {len(triples_with_missing_entities)} triples with missing entities from test to inference.")
    
    # Re-check
    test_entities = {entity for triple in final_test_set for entity in [triple[0], triple[2]]}
    inf_entities = {entity for triple in final_inf_graph for entity in [triple[0], triple[2]]}
    missing_entities = test_entities - inf_entities
    
    if not missing_entities:
        print("Successfully fixed missing entities issue.")
    else:
        print(f"WARNING: Still have {len(missing_entities)} missing entities.")

# 13. Print final statistics
print("\nFinal split statistics:")
print(f"Inference graph: {len(final_inf_graph)} triples with {len(inf_entities)} entities and {len(inf_relations)} relations")
print(f"Test set: {len(final_test_set)} triples with {len(test_entities)} entities and {len(test_relations)} relations")

if len(final_test_set) > 0:
    print(f"Ratio: {len(final_inf_graph) / len(final_test_set):.2f}:1")
else:
    print("Ratio: N/A (test set is empty)")

# Check relation exclusivity
relation_overlap = {r for _, r, _ in final_inf_graph} & {r for _, r, _ in final_test_set}
print(f"Relations overlap between sets: {len(relation_overlap)} relations")

# Check entity coverage
entity_coverage = len(test_entities & inf_entities) / len(test_entities) if test_entities else 0
print(f"Entity coverage: {entity_coverage:.2%} of test entities exist in inference graph")

# Check head relation distribution in test set
test_head_to_relations = defaultdict(set)
for h, r, _ in final_test_set:
    test_head_to_relations[h].add(r)

relation_counts = [len(relations) for relations in test_head_to_relations.values()]
if relation_counts:
    avg_relations_per_head = sum(relation_counts) / len(relation_counts)
    min_relations = min(relation_counts) if relation_counts else 0
    max_relations = max(relation_counts) if relation_counts else 0
    print(f"Test set head statistics: {len(test_head_to_relations)} heads, " 
          f"avg {avg_relations_per_head:.1f} relations per head (min: {min_relations}, max: {max_relations})")

# Count heads by relation count
head_relation_counts = defaultdict(int)
for h, relations in test_head_to_relations.items():
    head_relation_counts[len(relations)] += 1

print("\nDistribution of relation counts per head in test set:")
for relation_count in sorted(head_relation_counts.keys()):
    heads_count = head_relation_counts[relation_count]
    print(f"  {relation_count} relations: {heads_count} heads")

# 14. Write the final files
write_triples(new_inf_graph_path, final_inf_graph)
write_triples(new_inf_test_path, final_test_set)

print("\nSplit generation complete.")
print(f"New inference graph: {new_inf_graph_path}")
print(f"New test file: {new_inf_test_path}")