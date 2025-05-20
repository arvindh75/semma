import os
import random
from collections import defaultdict

script_dir = os.path.dirname(__file__)
raw_dir = os.path.join(script_dir, "NLIngram/kg-datasets/ingram/nl/100/raw")

original_inf_graph_path = os.path.join(raw_dir, "inference_graph.txt")
original_inf_test_path = os.path.join(raw_dir, "inf_test.txt")

new_inf_graph_path = os.path.join(raw_dir, "inference_graph_new.txt")
new_inf_test_path = os.path.join(raw_dir, "inf_test_new.txt")

# Fraction of relations to keep in inference graph
INFERENCE_RELATION_FRACTION = 0.8  # We'll keep 80% of relations in inference graph

# Minimum number of distinct relation types an entity must be involved with in the final test set
MIN_DISTINCT_RELATIONS_PER_TEST_ENTITY = 3

MIN_DISTINCT_RELATIONS_PER_TEST_HEAD = 3
MAX_ITERATIONS_COMPLEX_SPLIT = 100 # Max iterations for the new refinement loop

def read_triples_from_file(filepath):
    """Reads triples from a file into a set of (head, relation, tail) tuples."""
    triples = set()
    relations = set()
    print(f"Reading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    triples.add(tuple(parts)) # Store as tuple
                # else:
                #     print(f"Skipping malformed line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}. Returning empty set.")
        return set()
    print(f"Found {len(triples)} unique triples in {filepath}.")
    return triples

def write_triples(filepath, triples):
    """Writes triples to a file, one per line."""
    print(f"Writing {len(triples)} triples to {filepath}...")
    with open(filepath, 'w') as f:
        # Sort for deterministic output (optional, but good practice)
        sorted_triples = sorted(list(triples))
        for h, r, t in sorted_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    print("Writing complete.")


print("--- Loading Original Data ---")
original_triples_inf = read_triples_from_file(original_inf_graph_path)
original_triples_test = read_triples_from_file(original_inf_test_path)
all_unique_triples = original_triples_inf.union(original_triples_test)

print(f"Total unique triples combined: {len(all_unique_triples)}")

if not all_unique_triples:
    print("No triples found to process. Writing empty files.")
    write_triples(new_inf_graph_path, set())
    write_triples(new_inf_test_path, set())
    exit()

current_test_triples = all_unique_triples.copy()
final_test_triples = set() # To store the result

print(f"\n--- Starting Iterative Refinement for Test Set (Max Iterations: {MAX_ITERATIONS_COMPLEX_SPLIT}) ---")
print(f"Initial candidate test triples: {len(current_test_triples)}")

for iteration in range(MAX_ITERATIONS_COMPLEX_SPLIT):
    num_test_triples_before_iteration = len(current_test_triples)
    print(f"\nIteration {iteration + 1}/{MAX_ITERATIONS_COMPLEX_SPLIT}, Current test triples: {num_test_triples_before_iteration}")

    if not current_test_triples:
        print("No candidate test triples left. Stopping refinement.")
        break

    # a. Determine Implied Inference Graph based on current_test_triples
    R_current_test_relations = {r for _, r, _ in current_test_triples}
    # print(f"  Relations in current test set ({len(R_current_test_relations)}): {list(R_current_test_relations)[:10]}...") # Optional: for debugging
    
    current_inf_graph_triples = {(h,r,t) for h,r,t in all_unique_triples if r not in R_current_test_relations}
    E_current_inf_entities = {entity for h_inf, _, t_inf in current_inf_graph_triples for entity in (h_inf, t_inf)}
    # print(f"  Implied inference triples: {len(current_inf_graph_triples)}, Entities in implied inference: {len(E_current_inf_entities)}")

    # b. Filter 1: Entity Coverage for Test Triples
    # Entities in test triples must be present in the implied inference graph
    triples_to_keep_after_entity_filter = set()
    removed_by_entity_coverage = 0
    if not E_current_inf_entities and current_test_triples : # Optimization: if inf graph is empty, all test triples fail this
        print(f"  Filter 1 (Entity Coverage): Implied inference graph has no entities. All {len(current_test_triples)} current test triples will be removed.")
        removed_by_entity_coverage = len(current_test_triples)
        current_test_triples = set()
    else:
        for h_test, r_test, t_test in current_test_triples:
            if h_test in E_current_inf_entities and t_test in E_current_inf_entities:
                triples_to_keep_after_entity_filter.add((h_test, r_test, t_test))
            else:
                removed_by_entity_coverage +=1
        
        if removed_by_entity_coverage > 0:
            print(f"  Filter 1 (Entity Coverage): Removed {removed_by_entity_coverage} triples. {len(triples_to_keep_after_entity_filter)} remaining.")
            current_test_triples = triples_to_keep_after_entity_filter
            # If triples were removed by this filter, the basis for R_current_test_relations and E_current_inf_entities changes.
            # It's often better to restart or re-evaluate. Here, we continue to the next iteration's recalculation.
            if not current_test_triples: # Check if it became empty
                 print("  Filter 1 made test set empty. Stopping refinement.")
                 break
            # No 'continue' here; allow Filter 2 to run on the reduced set in the same iteration,
            # as Filter 2 might further refine it based on the *new* current_test_triples.
            # The loop structure will re-evaluate R_current_test_relations at the start of the next iteration.

    if not current_test_triples: # Re-check after Filter 1
        print("  No candidate test triples left after entity coverage filter.")
        break
    
    # c. Filter 2: Minimum Distinct Relations for Test Heads
    head_distinct_relations_in_test = defaultdict(set)
    all_heads_in_current_test = set()
    for h_trip, r_trip, _ in current_test_triples: # t_trip not needed here
        head_distinct_relations_in_test[h_trip].add(r_trip)
        all_heads_in_current_test.add(h_trip)

    low_connectivity_heads = {
        h for h in all_heads_in_current_test if len(head_distinct_relations_in_test[h]) < MIN_DISTINCT_RELATIONS_PER_TEST_HEAD
    }

    removed_by_min_relations = 0
    if low_connectivity_heads:
        # print(f"  Filter 2 (Min Head Relations): Found {len(low_connectivity_heads)} low-connectivity heads.")
        triples_to_keep_after_head_filter = set()
        for h_trip, r_trip, t_trip in current_test_triples:
            if h_trip not in low_connectivity_heads:
                triples_to_keep_after_head_filter.add((h_trip, r_trip, t_trip))
            else:
                removed_by_min_relations +=1
        
        if removed_by_min_relations > 0:
            print(f"  Filter 2 (Min Head Relations): Removed {removed_by_min_relations} triples. {len(triples_to_keep_after_head_filter)} remaining.")
            current_test_triples = triples_to_keep_after_head_filter
    
    # d. Check for Convergence
    if len(current_test_triples) == num_test_triples_before_iteration and removed_by_entity_coverage == 0 and removed_by_min_relations == 0 :
        print(f"\nConverged after {iteration + 1} iterations. Stable test set found.")
        break
    elif iteration == MAX_ITERATIONS_COMPLEX_SPLIT - 1:
        print(f"\nWarning: Reached max iterations ({MAX_ITERATIONS_COMPLEX_SPLIT}). The test set may not be fully stable.")

final_test_triples = current_test_triples
print(f"--- Refinement Complete ---")
print(f"Final selected test triples: {len(final_test_triples)}")

# 4. Define the new inference graph based on the final test set
R_final_test_relations = {r for _, r, _ in final_test_triples}
new_inf_graph_triples = {(h,r,t) for h,r,t in all_unique_triples if r not in R_final_test_relations}

# 5. Print final statistics
print("\n--- Final Split Statistics ---")
print(f"New Inference Graph Triples: {len(new_inf_graph_triples)}")
print(f"New Test Triples: {len(final_test_triples)}")

if len(final_test_triples) > 0 and len(new_inf_graph_triples) > 0 :
    # Sanity check: verify relation disjointness (optional)
    relations_in_inf = {r for _,r,_ in new_inf_graph_triples}
    common_relations = R_final_test_relations.intersection(relations_in_inf)
    if common_relations:
        print(f"Error: Found {len(common_relations)} common relations between final test and inference sets: {list(common_relations)[:5]}")
    else:
        print("Relation disjointness between test and inference sets confirmed.")
    
    # Sanity check: entity coverage for test set (optional)
    entities_in_final_inf = {e for h,_,t in new_inf_graph_triples for e in (h,t)}
    missing_entity_coverage_count = 0
    for h_f,_,t_f in final_test_triples:
        if not(h_f in entities_in_final_inf and t_f in entities_in_final_inf):
            missing_entity_coverage_count +=1
    if missing_entity_coverage_count > 0:
         print(f"Error: {missing_entity_coverage_count} test triples failed final entity coverage check.")
    else:
        print("Entity coverage for test triples confirmed.")

    # Sanity check: head distinct relations (optional)
    final_head_distinct_relations = defaultdict(set)
    final_test_heads = set()
    for h_f,r_f,_ in final_test_triples:
        final_head_distinct_relations[h_f].add(r_f)
        final_test_heads.add(h_f)
    
    failed_head_connectivity_count = 0
    for head_f in final_test_heads:
        if len(final_head_distinct_relations[head_f]) < MIN_DISTINCT_RELATIONS_PER_TEST_HEAD:
            failed_head_connectivity_count +=1
    if failed_head_connectivity_count > 0:
        print(f"Error: {failed_head_connectivity_count} heads in final test set failed min distinct relations check.")
    else:
        print("Minimum distinct relations for test heads confirmed.")

    # Ratio
    print(f"Resulting Ratio (Inf/Test): {len(new_inf_graph_triples)/float(len(final_test_triples)):.2f}:1" if len(final_test_triples) > 0 else "Test set is empty, ratio undefined.")

# 6. Write the new files
write_triples(new_inf_graph_path, new_inf_graph_triples)
write_triples(new_inf_test_path, final_test_triples)

print("\nTriple shuffling with new logic complete.")
print(f"New inference graph: {new_inf_graph_path}")
print(f"New test file: {new_inf_test_path}")