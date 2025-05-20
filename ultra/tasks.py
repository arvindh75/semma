from functools import reduce
from torch_scatter import scatter_add
from torch_geometric.data import Data
import torch
import re
import os
import numpy as np
import json

from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from ultra import parse

mydir = os.getcwd()
flags = parse.load_flags(os.path.join(mydir, "flags.yaml"))

def edge_match(edge_index, query_index):
    # O((n + q)logn) time
    # O(n) memory
    # edge_index: big underlying graph
    # query_index: edges to match

    # preparing unique hashing of edges, base: (max_node, max_relation) + 1
    base = edge_index.max(dim=1)[0] + 1
    # we will map edges to long ints, so we need to make sure the maximum product is less than MAX_LONG_INT
    # idea: max number of edges = num_nodes * num_relations
    # e.g. for a graph of 10 nodes / 5 relations, edge IDs 0...9 mean all possible outgoing edge types from node 0
    # given a tuple (h, r), we will search for all other existing edges starting from head h
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    # hash both the original edge index and the query index to unique integers
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern for each query in the batch
    num_match = end - start

    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)

    return order[range], num_match

def negative_sampling(data, batch, num_negative, strict=True):
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # strict negative sampling vs random negative sampling
    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[:batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        # draw samples for negative tails
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2:]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        # draw samples for negative heads
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(data.num_nodes, (batch_size, num_negative), device=batch.device)
        neg_t_index, neg_h_index = neg_index[:batch_size // 2], neg_index[batch_size // 2:]

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[:batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2:, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)

def all_negative(data, batch):
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    # generate all negative tails for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")  # indexing "xy" would return transposed
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    # generate all negative heads for this batch
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch

def strict_negative_mask(data, batch):
    # this function makes sure that for a given (h, r) batch we will NOT sample true tails as random negatives
    # similarly, for a given (t, r) we will NOT sample existing true heads as random negatives

    pos_h_index, pos_t_index, pos_r_index = batch.t()

    # part I: sample hard negative tails
    # edge index of all (head, relation) edges from the underlying graph
    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    # edge index of current batch (head, relation) for which we will sample negatives
    query_index = torch.stack([pos_h_index, pos_r_index])
    # search for all true tails for the given (h, r) batch
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(num_t_truth)
    t_mask = torch.ones(len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true tails
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    # part II: sample hard negative heads
    # edge_index[1] denotes tails, so the edge index becomes (t, r)
    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    # edge index of current batch (tail, relation) for which we will sample heads
    query_index = torch.stack([pos_t_index, pos_r_index])
    # search for all true heads for the given (t, r) batch
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    # build an index from the found edges
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(num_h_truth)
    h_mask = torch.ones(len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device)
    # assign 0s to the mask with the found true heads
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask

def compute_ranking(pred, target, mask=None):
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        # filtered ranking
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        # unfiltered ranking
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking

def get_relation_embeddings(relation_names, model_embed = None):
    
    device = torch.device(f"cuda:{flags.gpus}" if torch.cuda.is_available() else "cpu")

    if(model_embed == "sentbert"):
        # Load pre-trained language model and ensure it's on the specified device
        sbert_model = SentenceTransformer('all-mpnet-base-v2', device=device)
        embeddings = sbert_model.encode(
            relation_names, 
            convert_to_tensor=True  # Output tensor should be on `device`
        )

    else: # Assuming jinaai or other models
        model_hf_name = "jinaai/jina-embeddings-v3" # Default for this branch
        model = AutoModel.from_pretrained(model_hf_name, trust_remote_code=True)
        model.to(device)
        # model.encode for jinaai typically returns numpy arrays
        raw_embeddings = model.encode(relation_names, task="text-matching", truncate_dim=64)
        
        if isinstance(raw_embeddings, np.ndarray):
            embeddings = torch.from_numpy(raw_embeddings)
        elif isinstance(raw_embeddings, list) and len(raw_embeddings) > 0 and isinstance(raw_embeddings[0], np.ndarray):
            # Handle list of numpy arrays by stacking, e.g. if batch processing in encode returned list
            embeddings = torch.from_numpy(np.stack(raw_embeddings))
        elif isinstance(raw_embeddings, torch.Tensor):
            embeddings = raw_embeddings # Already a tensor
        else:
            raise TypeError(f"Unexpected embedding type from model.encode: {type(raw_embeddings)}")

    # Ensure the final embeddings tensor is on the correct device
    if not isinstance(embeddings, torch.Tensor):
        # This might happen if the model.encode path for 'else' resulted in an unexpected type
        # that wasn't converted to a tensor above.
        # For safety, convert if it's a common type like numpy array, otherwise raise error.
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        else:
            raise TypeError(f"Embeddings are not a PyTorch Tensor before final device placement: {type(embeddings)}")

    if embeddings.device != device:
        embeddings = embeddings.to(device)
        
    return embeddings

def find_top_k_similar_relations(embeddings, k, relation_names):
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    top_k_similarities = []

    for i in range(similarity_matrix.size(0)):
        top_k_with_self = similarity_matrix[i].topk(k + 1)
        indices = top_k_with_self.indices.tolist()  # Indices of top-k+1 similarities
        values = top_k_with_self.values.tolist()    # Similarity values (if needed)

        # Filter out self-similarity (where index == i)
        top_k = [(j, values[indices.index(j)]) for j in indices if j != i]

        # Collect the results
        top_k_similarities.extend([(i, j) for j, _ in top_k])

    return top_k_similarities  # List of (relation_i, relation_j) pairs

def find_top_x_percent(embeddings, relation_names):
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
 
    # Mask the diagonal (self-similarity) by setting it to -inf
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
 
    # Flatten the matrix and get all valid pairs with their similarity
    num_relations = similarity_matrix.size(0)
    total_possible_pairs = (num_relations * (num_relations - 1)) / 2  # Ordered pairs
 
    indices = torch.triu_indices(num_relations, num_relations, offset=1)
    similarities = similarity_matrix[indices[0], indices[1]]
 
    # Calculate number of pairs to select (top X%)
    here = flags.topx / 100
    num_pairs = int(total_possible_pairs * here)
    num_pairs = min(num_pairs, similarities.numel())
 
    top_similarities, top_indices = torch.topk(similarities, num_pairs)
 
    # Convert back to relation index pairs
    top_x_pairs = [(indices[0][i].item(), indices[1][i].item()) for i in top_indices]
    # add the inverse edges to the pairs
    top_x_pairs_inv = [(indices[1][i].item(), indices[0][i].item()) for i in top_indices]
    top_x_pairs.extend(top_x_pairs_inv)
    
    print("====================================")
    print("Number of similar relation pairs: ", len(top_x_pairs))
    
    # Uncomment if you want to print the actual relations
    # for i, (rel1, rel2) in enumerate(top_x_pairs):
    #     if rel1 < num_relations // 2 and rel2 < num_relations // 2:
    #         print(relation_names[rel1], "---", relation_names[rel2])
 
    return top_x_pairs

def find_pairs_above_threshold(embeddings, relation_names):
    # Compute the cosine similarity matrix
    similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    # Mask the diagonal (self-similarity)
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    num_relations = similarity_matrix.size(0)
    row_indices, col_indices = torch.where(similarity_matrix > flags.threshold)
    
    # Convert to relation index pairs
    selected_pairs = [(row.item(), col.item()) for row, col in zip(row_indices, col_indices)]
    
    print("====================================")
    print(f"Number of relation pairs with cosine similarity > {flags.threshold}: {len(selected_pairs)}")
    # Uncomment if you want to print the actual relations
    # for row, col in zip(row_indices, col_indices):
    #     rel1, rel2 = row.item(), col.item()
    #     if(rel1 < num_relations // 2 and rel2 < num_relations // 2):
    #         sim_value = similarity_matrix[rel1, rel2].item()
    #         print(relation_names[rel1], "---", relation_names[rel2], "---", sim_value)

    return selected_pairs

def load_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def create_harder_rg1(graph, original_rel_graph, test_relation, inverse_relation, head=None, tail=None):
    
    device = graph.edge_index.device
    num_rels = graph.num_relations
    
    # Copy the original relation graph edges
    new_edge_index = original_rel_graph.edge_index.clone()
    new_edge_type = original_rel_graph.edge_type.clone()
    
    test_rel_node = test_relation  # New node ID
    inverse_rel_node = inverse_relation  # New node ID
    
    # Lists to collect new edges and types
    new_edges = []
    new_types = []
    
    # Case 1: Only head is known
    if head is not None and tail is None:
        # Find all relations where this entity appears as head (h2h connections)
        head_mask = (graph.edge_index[0] == head)
        head_relations = graph.edge_type[head_mask].unique()
        
        # Add h2h connections
        for rel in head_relations:
            if rel != test_relation:  # Avoid self-loops
                new_edges.append([test_rel_node, rel.item()])
                new_types.append(0)  # h2h type
                
                # Reverse connection: rel -> test_relation (h2h)
                new_edges.append([rel.item(), test_rel_node])
                new_types.append(0)  # h2h type
        
        # Find all relations where this entity appears as tail (t2h connections)
        tail_mask = (graph.edge_index[1] == head)
        tail_relations = graph.edge_type[tail_mask].unique()
        
        # Add t2h connections
        for rel in tail_relations:
            new_edges.append([test_rel_node, rel.item()])
            new_types.append(2)  # h2t type
            
            # Reverse connection: rel -> test_relation (t2h)
            new_edges.append([rel.item(), test_rel_node])
            new_types.append(3)  # t2h type

        # For the inverse relation (tail is known for the inverse)
        # For inverse relation, the head in original becomes tail
        # t2h connections for inverse relation
        for rel in head_relations:
            new_edges.append([inverse_rel_node, rel.item()])
            new_types.append(3)  # t2h type
            
            new_edges.append([rel.item(), inverse_rel_node])
            new_types.append(2)  # h2t type
        
        # t2t connections for inverse relation
        for rel in tail_relations:
            if rel != inverse_relation:  # Avoid self-loops
                new_edges.append([inverse_rel_node, rel.item()])
                new_types.append(1)  # t2t type
                
                # Reverse connection: rel -> inverse_relation (h2t)
                new_edges.append([rel.item(), inverse_rel_node])
                new_types.append(1)  # t2t type
        
        # Connect test relation with its inverse
        new_edges.append([test_rel_node, inverse_rel_node])
        new_types.append(2)  # h2t type - original to inverse
        
        new_edges.append([inverse_rel_node, test_rel_node])
        new_types.append(3)  # t2h type - inverse to original
    
    # Case 2: Only tail is known
    elif tail is not None and head is None:
        
        tail_mask = (graph.edge_index[1] == tail)
        tail_relations = graph.edge_type[tail_mask].unique()
        
        # Add t2t connections
        for rel in tail_relations:
            if rel != test_relation:  # Avoid self-loops
                new_edges.append([test_rel_node, rel.item()])
                new_types.append(1)  # t2t type
                
                # Reverse connection: rel -> test_relation (t2t)
                new_edges.append([rel.item(), test_rel_node])
                new_types.append(1)  # t2t type
        
        # Find all relations where this entity appears as head (h2t connections)
        head_mask = (graph.edge_index[0] == tail)
        head_relations = graph.edge_type[head_mask].unique()
        
        # Add h2t connections
        for rel in head_relations:
            new_edges.append([test_rel_node, rel.item()])
            new_types.append(3)  # t2h type
            
            # Reverse connection: rel -> test_relation (h2t)
            new_edges.append([rel.item(), test_rel_node])
            new_types.append(2)  # h2h type
    
        # For the inverse relation (head is known for the inverse)
        # For inverse relation, the tail in original becomes head
        # h2h connections for inverse relation
        for rel in tail_relations:
            new_edges.append([inverse_rel_node, rel.item()])
            new_types.append(2)  # h2t type
            
            # Reverse connection: rel -> inverse_relation (h2h)
            new_edges.append([rel.item(), inverse_rel_node])
            new_types.append(3)  # t2h type
        
        # t2h connections for inverse relation
        for rel in head_relations:
            if rel != inverse_relation:  # Avoid self-loops
                new_edges.append([inverse_rel_node, rel.item()])
                new_types.append(0)  # h2h type
            
                # Reverse connection: rel -> inverse_relation (t2h)
                new_edges.append([rel.item(), inverse_rel_node])
                new_types.append(0)  # h2h type
        
        # Connect test relation with its inverse
        new_edges.append([test_rel_node, inverse_rel_node])
        new_types.append(2)  # h2t type - original to inverse
        
        new_edges.append([inverse_rel_node, test_rel_node])
        new_types.append(3)  # t2h type - inverse to original
    
    # Convert to tensors if there are new edges
    if new_edges:
        new_edge_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).T
        new_type_tensor = torch.tensor(new_types, dtype=torch.long, device=device)
        
        # Add to existing edges
        new_edge_index = torch.cat([new_edge_index, new_edge_tensor], dim=1)
        new_edge_type = torch.cat([new_edge_type, new_type_tensor])
    
    # Create the new relation graph
    test_rel_graph = Data(
        edge_index=new_edge_index,
        edge_type=new_edge_type,
        num_nodes=num_rels,  # Original relations + test relation + inverse test relation
        num_relations=4  # Same relation types as before
    )

    return test_rel_graph

def build_relation_graph(graph):

    # expect the graph is already with inverse edges

    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)
    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head

    rel_graph = Data(
        edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
        edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
        num_nodes=num_rels, 
        num_relations=4
    )

    graph.relation_graph = rel_graph

    if(flags.harder_setting == True):
        if hasattr(graph, "is_harder") and graph.is_harder == True:
            graph.harder_head_rg1 = {}
            graph.harder_tail_rg1 = {}
            for i in range(graph.target_edge_index.size(1)):
                head = graph.target_edge_index[0, i].item()
                relation = num_rels // 2 - 1
                tail = graph.target_edge_index[1, i].item()

                inverse_relation = relation + (num_rels // 2) 
                
                # Create a new relation graph with the test relation as a new node
                # For head known case
                head_known_rel_graph = create_harder_rg1(
                    graph, rel_graph, relation, inverse_relation, head=head, tail=None
                )
                
                # For tail known case
                tail_known_rel_graph = create_harder_rg1(
                    graph, rel_graph, relation, inverse_relation, head=None, tail=tail
                )
                
                # Store these relation graphs
                graph.harder_head_rg1[(head, graph.target_edge_type[i].item())] = head_known_rel_graph
                graph.harder_tail_rg1[(tail, graph.target_edge_type[i].item())] = tail_known_rel_graph
    
    return graph

def order_embeddings(embeddings, relation_names, graph, num_rels, inv_embeddings = None):
    ordered_embeddings = {}
    for i in range(len(embeddings)):
        if(relation_names[i] in graph.edge2id):
            ordered_embeddings[graph.edge2id[relation_names[i]]] = embeddings[i]
            if inv_embeddings is not None:
                ordered_embeddings[graph.edge2id[relation_names[i]] + len(graph.edge2id)] = inv_embeddings[i]
            else:
                ordered_embeddings[graph.edge2id[relation_names[i]] + len(graph.edge2id)] = -embeddings[i]

    embeddings = []
    for i in range(2*len(graph.edge2id)):
        embeddings.append(ordered_embeddings[i])

    return embeddings

def build_relation_graph_exp(graph):
    # Extract existing edge indices and types
    k = flags.k
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dh = scatter_add(torch.ones_like(Eh[:, 1]), Eh[:, 0])

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T, 
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]], 
        (num_rels, num_nodes)
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, 
        torch.ones(Eh.shape[0], device=device), 
        (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)  # (num_edges, 2)

    Dt = scatter_add(torch.ones_like(Et[:, 1]), Et[:, 0])
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T, 
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]], 
        (num_rels, num_nodes)
    )
    Et = torch.sparse_coo_tensor(
        Et.T, 
        torch.ones(Et.shape[0], device=device), 
        (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat([Ahh.indices().T, torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0)], dim=1)  # head to head
    tt_edges = torch.cat([Att.indices().T, torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1)], dim=1)  # tail to tail
    ht_edges = torch.cat([Aht.indices().T, torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2)], dim=1)  # head to tail
    th_edges = torch.cat([Ath.indices().T, torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3)], dim=1)  # tail to head

    # Step 4: Add fifth edge type for semantically similar relations
    # Generate embeddings for relation names based on the edge2id dictionary keys

    if(flags.run == "semma"):
        rel_graph = Data(
            edge_index=torch.cat([hh_edges[:, [0, 1]].T, tt_edges[:, [0, 1]].T, ht_edges[:, [0, 1]].T, th_edges[:, [0, 1]].T], dim=1), 
            edge_type=torch.cat([hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0),
            num_nodes=num_rels, 
            num_relations=4
        )
        graph.relation_graph = rel_graph

        if(flags.harder_setting == True):
            if hasattr(graph, "is_harder") and graph.is_harder == True:
                graph.harder_head_rg1 = {}
                graph.harder_tail_rg1 = {}
                for i in range(graph.target_edge_index.size(1)):
                    head = graph.target_edge_index[0, i].item()
                    relation = num_rels // 2 - 1
                    tail = graph.target_edge_index[1, i].item()

                    inverse_relation = relation + (num_rels // 2) 
                    
                    # Create a new relation graph with the test relation as a new node
                    # For head known case
                    head_known_rel_graph = create_harder_rg1(
                        graph, rel_graph, relation, inverse_relation, head=head, tail=None
                    )
                    
                    # For tail known case
                    tail_known_rel_graph = create_harder_rg1(
                        graph, rel_graph, relation, inverse_relation, head=None, tail=tail
                    )
                    
                    # Store these relation graphs
                    graph.harder_head_rg1[(head, graph.target_edge_type[i].item())] = head_known_rel_graph
                    graph.harder_tail_rg1[(tail, graph.target_edge_type[i].item())] = tail_known_rel_graph
    
    file_path = None

    if flags.LLM == "gpt4o":
        file_path = os.path.join(mydir, "openrouter/descriptions/gpt-4o-2024-11-20", graph.dataset + ".json")
    elif flags.LLM == "qwen3-32b":
        file_path = os.path.join(mydir, "openrouter/descriptions/qwen3-32b", graph.dataset + ".json")
    elif flags.LLM == "deepseekv3":
        file_path = os.path.join(mydir, "openrouter/descriptions/deepseek-chat-v3-0324", graph.dataset + ".json")

    if file_path is not None:
        dict = load_file(file_path)
    else:
        raise ValueError("LLM not supported")

    cleaned_relations = dict['cleaned_relations']
    relation_descriptions = dict['relation_descriptions']

    relation_names = list(cleaned_relations.keys())

    cleaned_relations = list(cleaned_relations.values())
    relation_descriptions = list(relation_descriptions.values())
    # relation_descriptions[i][0] -> description of relation i and [i][1] -> description of inverse of relation i
    rel_desc = []
    for i in range(len(relation_descriptions)):
        rel_desc.append(relation_descriptions[i][0])
    inv_desc = []
    for i in range(len(relation_descriptions)):
        inv_desc.append(relation_descriptions[i][1])

    # traverse through edge_type and get the relation names

    if(flags.rg2_embedding == "combined"):
        embeddings_nollm = get_relation_embeddings(relation_names, flags.model_embed)
        embeddings_llm_name = get_relation_embeddings(cleaned_relations, flags.model_embed)
        embeddings_llm_desc = get_relation_embeddings(rel_desc, flags.model_embed)
        embeddings_llm_inv_desc = get_relation_embeddings(inv_desc, flags.model_embed)
        embeddings_nollm = order_embeddings(embeddings_nollm, relation_names, graph, num_rels)
        embeddings_llm_name = order_embeddings(embeddings_llm_name, relation_names, graph, num_rels)
        embeddings_llm_desc = order_embeddings(embeddings_llm_desc, relation_names, graph, num_rels, embeddings_llm_inv_desc)
        embeddings = [
            (a + b + c) / 3
            for a, b, c in zip(embeddings_nollm, embeddings_llm_name, embeddings_llm_desc)
        ]

    elif(flags.rg2_embedding == "combined-sum"):
        embeddings_nollm = get_relation_embeddings(relation_names, flags.model_embed)
        embeddings_llm_name = get_relation_embeddings(cleaned_relations, flags.model_embed)
        embeddings_llm_desc = get_relation_embeddings(rel_desc, flags.model_embed)
        embeddings_llm_inv_desc = get_relation_embeddings(inv_desc, flags.model_embed)
        embeddings_nollm = order_embeddings(embeddings_nollm, relation_names, graph, num_rels)
        embeddings_llm_name = order_embeddings(embeddings_llm_name, relation_names, graph, num_rels)
        embeddings_llm_desc = order_embeddings(embeddings_llm_desc, relation_names, graph, num_rels, embeddings_llm_inv_desc)
        embeddings = [
            (a + b + c)
            for a, b, c in zip(embeddings_nollm, embeddings_llm_name, embeddings_llm_desc)
        ]

    elif(flags.rg2_embedding == "no llm"):
        embeddings = get_relation_embeddings(relation_names, flags.model_embed)
        embeddings = order_embeddings(embeddings, relation_names, graph, num_rels)
    elif(flags.rg2_embedding == "llm name"):
        embeddings = get_relation_embeddings(cleaned_relations, flags.model_embed)
        embeddings = order_embeddings(embeddings, relation_names, graph, num_rels)
    elif(flags.rg2_embedding == "llm description"):
        embeddings = get_relation_embeddings(rel_desc, flags.model_embed) # (num_relations, embedding size)
        inv_embeddings = get_relation_embeddings(inv_desc, flags.model_embed)
        embeddings = order_embeddings(embeddings, relation_names, graph, num_rels, inv_embeddings)

    if flags.harder_setting == True:
        if hasattr(graph, "is_harder") and graph.is_harder == True:            
            graph.harder_head_rg2 = {}
            graph.harder_tail_rg2 = {}
            for i in range(graph.target_edge_index.size(1)): 
                head = graph.target_edge_index[0, i].item()
                relation = num_rels // 2 - 1
                tail = graph.target_edge_index[1, i].item()

                inverse_relation = relation + (num_rels // 2)

                embeddings_here = []
                for j in range(num_rels // 2 - 1):
                    embeddings_here.append(embeddings[j])
                embeddings_here.append(embeddings[graph.target_edge_type[i].item()])
                for j in range(num_rels // 2 - 1):
                    embeddings_here.append(embeddings[j + len(graph.edge2id)])
                embeddings_here.append(embeddings[graph.target_edge_type[i].item() + len(graph.edge2id)])
                
                embeddings_here = torch.stack(embeddings_here).to(device)
                relation_similarity_matrix = F.cosine_similarity(embeddings_here.unsqueeze(1), embeddings_here.unsqueeze(0), dim=2)
                
                if(flags.k != 0):
                    similar_pairs = find_top_k_similar_relations(embeddings_here, k, relation_names)
                elif(flags.topx != 0):
                    similar_pairs = find_top_x_percent(embeddings_here, relation_names)
                else:
                    similar_pairs = find_pairs_above_threshold(embeddings_here, relation_names)

                if len(similar_pairs) == 0:
                    sem_sim_edges = torch.empty((0, 2), dtype=torch.long, device=device)
                    sem_sim_type = torch.empty((0,), dtype=torch.long, device=device)
                else:
                    # Add new edges for semantically similar relations with type 0
                    sem_sim_edges = torch.tensor(similar_pairs, dtype=torch.long, device=device)
                    sem_sim_type = torch.full((sem_sim_edges.shape[0],), 0, dtype=torch.long, device=device)
                    
                if flags.use_cos_sim_weights:
                    rel_graph2_here = Data(
                        edge_index=sem_sim_edges.T,
                        edge_type=sem_sim_type,
                        num_nodes=num_rels,
                        num_relations=1,  
                        relation_embeddings = embeddings_here,
                        relation_similarity_matrix = relation_similarity_matrix
                    )
                else:
                    rel_graph2_here = Data(
                        edge_index=sem_sim_edges.T,
                        edge_type=sem_sim_type,
                        num_nodes=num_rels,
                        num_relations=1, 
                        relation_embeddings = embeddings_here
                    )
                graph.harder_head_rg2[(head, graph.target_edge_type[i].item())] = rel_graph2_here
                graph.harder_tail_rg2[(tail, graph.target_edge_type[i].item())] = rel_graph2_here
            
    embeddings = torch.stack(embeddings).to(device)
    # Calculate full cosine similarity matrix
    relation_similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

    # Find top-k similar relations
    if(flags.k != 0):
        similar_pairs = find_top_k_similar_relations(embeddings, k, relation_names)
    elif(flags.topx != 0):
        similar_pairs = find_top_x_percent(embeddings, relation_names)
    else:
        similar_pairs = find_pairs_above_threshold(embeddings, relation_names)

    if len(similar_pairs) == 0:
        sem_sim_edges = torch.empty((0, 2), dtype=torch.long, device=device)
        sem_sim_type = torch.empty((0,), dtype=torch.long, device=device)
    else:
        # Add new edges for semantically similar relations with type 0
        sem_sim_edges = torch.tensor(similar_pairs, dtype=torch.long, device=device)
        sem_sim_type = torch.full((sem_sim_edges.shape[0],), 0, dtype=torch.long, device=device)

    if flags.use_cos_sim_weights:
        rel_graph2 = Data(
            edge_index=sem_sim_edges.T,
            edge_type=sem_sim_type,
            num_nodes=num_rels,
            num_relations=1,  # Updated for 5 relation types
            relation_embeddings = embeddings,
            relation_similarity_matrix = relation_similarity_matrix
        )
    else:
        rel_graph2 = Data(
            edge_index=sem_sim_edges.T,
            edge_type=sem_sim_type,
            num_nodes=num_rels,
            num_relations=1,  # Updated for 5 relation types
            relation_embeddings = embeddings
        )
    graph.relation_graph2 = rel_graph2

    return graph