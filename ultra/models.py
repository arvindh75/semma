import torch
import os
from torch import nn
import torch.nn.functional as F # Added for softmax

from ultra.parse import parse_args
from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet
from ultra import parse 

mydir = os.getcwd()
flags = parse.load_flags(os.path.join(mydir, "flags.yaml"))

class CombineEmbeddings(nn.Module):
    def __init__(self, embedding_dim=64):
        super(CombineEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        
        # MLP layer for 'mlp' mode
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # if flags.embedding_combiner == 'attention':
        #     self.w = nn.Parameter(torch.Tensor(embedding_dim))
        #     nn.init.xavier_uniform_(self.w.unsqueeze(0)) # Initialize weights

    def forward(self, structural, semantic):
        
        if flags.embedding_combiner == 'mlp':
            combined = torch.cat([structural, semantic], dim=-1)
            return self.fc(combined)
            
        elif flags.embedding_combiner == 'concat':
            return torch.cat([structural, semantic], dim=-1)
            
        elif flags.embedding_combiner == 'attention':
            # v1: structural, v2: semantic
            # Both structural and semantic are expected to have shape (batch_size, num_nodes, embedding_dim)
            # w has shape (embedding_dim,)

            # Calculate dot products w^T v1 and w^T v2 for each node
            # Output shape: (batch_size, num_nodes)
            score1 = torch.einsum('bnd,d->bn', structural, self.w)
            score2 = torch.einsum('bnd,d->bn', semantic, self.w)

            # Stack scores for softmax: (batch_size, num_nodes, 2)
            scores = torch.stack([score1, score2], dim=2)

            # Calculate attention weights alpha1, alpha2 using softmax across the structural/semantic dimension
            # Output shape: (batch_size, num_nodes, 2)
            att_weights = F.softmax(scores, dim=2)

            # Extract alpha1 and alpha2, unsqueeze for broadcasting: (batch_size, num_nodes, 1)
            alpha1 = att_weights[:, :, 0].unsqueeze(-1)
            alpha2 = att_weights[:, :, 1].unsqueeze(-1)

            # Calculate the combined embedding: alpha1 * v1 + alpha2 * v2
            # Output shape: (batch_size, num_nodes, embedding_dim)
            combined_embedding = alpha1 * structural + alpha2 * semantic
            return combined_embedding
            
        else:
            raise ValueError(f"Unknown embedding_combiner mode: {flags.embedding_combiner}")

class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg, sem_model_cfg=None):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        # adding a bit more flexibility to initializing proper rel/ent classes from the configs
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)
        if(flags.run == "semma"):
            self.semantic_model = globals()[sem_model_cfg.pop('class')](**sem_model_cfg)
            # Initialize combiner, assuming input embeddings are 64-dimensional
            self.combiner = CombineEmbeddings(embedding_dim=64)
        self.relation_representations_structural = None
        self.relation_representations_semantic = None
        self.final_relation_representations = None

    def get_relation_representations(self):
        return self.relation_representations_structural, self.relation_representations_semantic, self.final_relation_representations

    def forward(self, data, batch, is_tail=False):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        # print("batch shape: ", batch[0])
        # exit()
        query_rels = batch[:, 0, 2]
        # print(query_rels)
        # exit()
        query_rels_traverse = batch[:, 0, :]
        # print("================================================")
        # print(query_rels_traverse)
        # print("================================================")
        # exit()
            
        if(flags.run == "semma"):
            if flags.harder_setting == True:
                relation_reprs = []
                if is_tail == True:
                    for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
                        # Use i-th query and i-th relation graph
                        # print(head, tail, query_rel)
                        # exit()
                        # convert head and query_rel to int
                        relation_graph = data.harder_tail_rg1[(int(tail), int(query_rel))]
                        # print(relation_graph)
                        # query_rels = query_rel.unsqueeze(0)  # shape [1]
                        # print(query_rel)
                        # exit()
                        query_rel = data.num_relations // 2 - 1
                        query_rel = torch.tensor([query_rel], device=batch.device)
                        out = self.relation_model(relation_graph, query=query_rel)  # shape [1, num_rels, hidden]
                        relation_reprs.append(out)
                        # print(head, query_rel, out.shape)
                        # exit()
                    # Stack to shape [batch_size, num_rels, hidden_dim]
                    self.relation_representations_structural = torch.cat(relation_reprs, dim=0)

                    # Semantic relation representations
                    semantic_relation_reprs = []
                    for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
                        relation_graph = data.harder_tail_rg2[(int(tail), int(query_rel))]
                        query_rel = data.num_relations // 2 - 1
                        query_rel = torch.tensor([query_rel], device=batch.device)
                        out = self.semantic_model(relation_graph, query=query_rel)  # shape [1, num_rels, hidden]
                        semantic_relation_reprs.append(out)
                        # print(head, query_rel, out.shape)
                        # exit()
                    self.relation_representations_semantic = torch.cat(semantic_relation_reprs, dim=0)
                else:
                    for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
                        # Use i-th query and i-th relation graph
                        # print(head, tail, query_rel)
                        # exit()
                        # convert head and query_rel to int
                        relation_graph = data.harder_head_rg1[(int(head), int(query_rel))]
                        # query_rels = query_rel.unsqueeze(0)  # shape [1]
                        # print("================================================")
                        # print(query_rel)
                        # exit()
                        query_rel = data.num_relations // 2 - 1
                        # print(query_rel)
                        # exit()
                        query_rel = torch.tensor([query_rel], device=batch.device)
                        out = self.relation_model(relation_graph, query=query_rel)  # shape [1, num_rels, hidden]
                        relation_reprs.append(out)
                        # print(head, query_rel, out.shape)
                        # exit()
                    # Stack to shape [batch_size, num_rels, hidden_dim]
                    # exit()
                    self.relation_representations_structural = torch.cat(relation_reprs, dim=0)

                    # Semantic relation representations
                    semantic_relation_reprs = []
                    for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
                        relation_graph = data.harder_head_rg2[(int(head), int(query_rel))]
                        # print(query_rel)
                        query_rel = data.num_relations // 2 - 1
                        # print(query_rel)
                        # exit()
                        query_rel = torch.tensor([query_rel], device=batch.device)
                        out = self.semantic_model(relation_graph, query=query_rel)  # shape [1, num_rels, hidden]
                        semantic_relation_reprs.append(out)
                        # print(head, query_rel, out.shape)
                        # exit()
                    self.relation_representations_semantic = torch.cat(semantic_relation_reprs, dim=0)
            else:
                self.relation_representations_structural = self.relation_model(data, query=query_rels)
                self.relation_representations_semantic = self.semantic_model(data, query=query_rels)

            self.final_relation_representations = self.combiner(self.relation_representations_structural, self.relation_representations_semantic)
            # print(final_relation_representations.shape)
            # exit()
            
        else:
            if flags.harder_setting == True:
                relation_reprs = []
                if is_tail == True:
                    for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
                        # Use i-th query and i-th relation graph
                        # print(head, tail, query_rel)
                        # exit()
                        # convert head and query_rel to int
                        relation_graph = data.harder_tail_rg1[(int(tail), int(query_rel))]
                        # print(relation_graph)
                        # query_rels = query_rel.unsqueeze(0)  # shape [1]
                        # print(query_rel)
                        # exit()
                        query_rel = data.num_relations // 2 - 1
                        query_rel = torch.tensor([query_rel], device=batch.device)
                        out = self.relation_model(relation_graph, query=query_rel)  # shape [1, num_rels, hidden]
                        relation_reprs.append(out)
                        # print(head, query_rel, out.shape)
                        # exit()
                    # Stack to shape [batch_size, num_rels, hidden_dim]
                    self.relation_representations_structural = torch.cat(relation_reprs, dim=0)
                else:
                    for i, (head, tail, query_rel) in enumerate(query_rels_traverse):
                        # Use i-th query and i-th relation graph
                        # print(head, tail, query_rel)
                        # exit()
                        # convert head and query_rel to int
                        relation_graph = data.harder_head_rg1[(int(head), int(query_rel))]
                        # query_rels = query_rel.unsqueeze(0)  # shape [1]
                        # print(query_rel)
                        # exit()
                        query_rel = data.num_relations // 2 - 1
                        query_rel = torch.tensor([query_rel], device=batch.device)
                        out = self.relation_model(relation_graph, query=query_rel)  # shape [1, num_rels, hidden]
                        relation_reprs.append(out)
                        # print(head, query_rel, out.shape)
                        # exit()
                    # Stack to shape [batch_size, num_rels, hidden_dim]
                    self.relation_representations_structural = torch.cat(relation_reprs, dim=0)
                self.final_relation_representations = self.relation_representations_structural
            else:
                self.final_relation_representations = self.relation_model(data, query=query_rels)
                self.relation_representations_structural = self.final_relation_representations

        # print("================================================")
        # print(self.relation_representations_structural.shape)
        # print("================================================")
        # exit()

        # if is_tail == True:
        #     print("================================================")
        #     print(self.final_relation_representations.shape)
        #     print("================================================")
        #     exit()
        
        score = self.entity_model(data, self.final_relation_representations, batch)
        # if is_tail == True:
        #     print("================================================")
        #     print(score.shape)
        #     print("================================================")
        #     exit()
        
        return score

# NBFNet to work on the graph of relations with 4/5 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):
    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):

        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, h_index, separate_grad=False, relation_embeddings=None):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        # print("====================================")
        # print("boundary shape: ", boundary.shape)
        # print("query shape: ", query.shape)
        # print(query)
        # print("edge_weight shape: ", edge_weight.shape)
        # print("embeddings of relation 0: ", boundary[0, 0])
        # exit()

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            # print(layer_input)
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            # print("====================================")
            # print(edge_weight)
            edge_weights.append(edge_weight)
            layer_input = hidden
        # exit()

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, graph, query):
        if flags.harder_setting == True:
            rel_graph = graph
        else:
            rel_graph = graph.relation_graph
        # message passing and updated node representations (that are in fact relations)
        if hasattr(rel_graph, "relation_embeddings"):
            output = self.bellmanford(rel_graph, h_index=query, relation_embeddings=rel_graph.relation_embeddings)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        else:
            output = self.bellmanford(rel_graph, h_index=query, relation_embeddings=None)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        return output
    
class SemRelNBFNet(BaseNBFNet):
    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

        if(flags.model_embed != "jinaai"):
            self.embedding_reducer = nn.Linear(768, 64)

    def bellmanford(self, data, h_index, separate_grad=False, relation_embeddings=None):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        embedding = relation_embeddings
        if(flags.model_embed != "jinaai"):
            reduced_embedding = self.embedding_reducer(embedding)
        else:
            reduced_embedding = embedding

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        
        # Create a mask for valid indices
        valid_indices = torch.arange(data.num_nodes, device=reduced_embedding.device)

        # Populate the boundary tensor for all batches in parallel
        boundary[:, valid_indices] = reduced_embedding.unsqueeze(0)
        # print("====================================")
        # print(valid_indices.shape)
        # print(valid_indices)
        # print(boundary.shape)
        # print(boundary[0][0])
        # print(boundary[0][11])

        # exit()
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes

        # for i in range(batch_size):
        #     for j in range(data.num_nodes):
        #         if j >= len(reduced_embedding):
        #             boundary[i, j] = (torch.zeros(len(reduced_embedding[0])))
        #             continue
        #         boundary[i, j] = reduced_embedding[j]

        size = (data.num_nodes, data.num_nodes)
        # Initialize edge weights using relation similarity if available
        if flags.use_cos_sim_weights and hasattr(data, "relation_similarity_matrix"):
            # Get source and target relation indices for each edge
            src, dst = data.edge_index
            # Look up similarity scores from the precomputed matrix
            edge_weight = data.relation_similarity_matrix[src, dst]
            # Ensure weights are on the correct device
            edge_weight = edge_weight.to(h_index.device)
            # Optional: Add a small epsilon or clamp to avoid zero weights if needed
            # edge_weight = torch.clamp(edge_weight, min=1e-6)
        else:
            # Default to ones if similarity matrix is not found
            edge_weight = torch.ones(data.num_edges, device=h_index.device)

        # print("====================================")
        # print(edge_weight.shape)
        # print(edge_weight)
        # exit()

        # print("====================================")
        # print("boundary shape: ", boundary.shape)
        # print("query shape: ", query.shape)
        # print(query)
        # print("edge_weight shape: ", edge_weight.shape)
        # print("embeddings of relation 0: ", boundary[0, 0])
        # exit()

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            # print(layer_input)
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            # print("====================================")
            # print(edge_weight)
            edge_weights.append(edge_weight)
            layer_input = hidden
        # exit()

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, graph, query):
        if flags.harder_setting == True:
            rel_graph = graph
        else:
            rel_graph = graph.relation_graph2
        # message passing and updated node representations (that are in fact relations)
        if hasattr(rel_graph, "relation_embeddings"):
            output = self.bellmanford(rel_graph, h_index=query, relation_embeddings=rel_graph.relation_embeddings)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        else:
            output = self.bellmanford(rel_graph, h_index=query, relation_embeddings=None)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        return output

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        # Initialize edge weights using relation similarity if available
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        if flags.harder_setting == True:
            # change r_index to be data.num_relations // 2 - 1
            r_index = torch.ones_like(r_index) * (data.num_relations // 2 - 1)

        # print("================================================")
        # print(h_index)
        # print(t_index)
        # print(r_index)
        # print("================================================")
        # exit()

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # print("================================================")
        # print(self.query.shape)
        # print("================================================")

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            # Ensure edge_index is sorted for rspmm after potential modification by remove_easy_edges
            # if flags.harder_setting == True and hasattr(data, "edge_index") and data.edge_index.numel() > 0:
            #         # N in rspmm.py's key calculation (edge_index[0] * N + edge_index[1])
            #     # corresponds to data.num_nodes because the input to rspmm has shape (batch_size, num_nodes, features).
            #     num_nodes_for_sort = data.num_nodes
            #     perm = (data.edge_index[0] * num_nodes_for_sort + data.edge_index[1]).argsort()
            #     data.edge_index = data.edge_index[:, perm]
            #     data.edge_type = dhata.edge_type[perm]
                # If data has other per-edge attributes that were modified/filtered by remove_easy_edges
                # and are used by rspmm or subsequent operations, they would also need this permutation.
                # Currently, edge_weight in bellmanford is initialized fresh after this block.

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

class QueryNBFNet(EntityNBFNet):
    """
    The entity-level reasoner for UltraQuery-like complex query answering pipelines
    Almost the same as EntityNBFNet except that 
    (1) we already get the initial node features at the forward pass time 
    and don't have to read the triples batch
    (2) we get `query` from the outer loop
    (3) we return a distribution over all nodes (assuming t_index = all nodes)
    """
    
    def bellmanford(self, data, node_features, query, separate_grad=False):
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, node_features, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, node_features, relation_representations, query):

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        # we already did traversal_dropout in the outer loop of UltraQuery
        # if self.training:
        #     # Edge dropout in the training mode
        #     # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
        #     # to make NBFNet iteration learn non-trivial paths
        #     data = self.remove_easy_edges(data, h_index, t_index, r_index)

        # node features arrive in shape (bs, num_nodes, dim)
        # NBFNet needs batch size on the first place
        output = self.bellmanford(data, node_features, query)  # (num_nodes, batch_size, feature_dim）
        score = self.mlp(output["node_feature"]).squeeze(-1) # (bs, num_nodes)
        return score  
