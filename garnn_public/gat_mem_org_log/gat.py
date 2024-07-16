from util import *

class GATLayer_exp(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection1 = nn.Linear(c_in, c_out * num_heads)
        self.projection2 = nn.Linear(c_in, c_out * num_heads)
        self.a_l = nn.Parameter(torch.Tensor(num_heads, c_out)) # One per head
        self.a_r = nn.Parameter(torch.Tensor(num_heads, c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        # nn.init.orthogonal_(self.projection.weight.data)
        # nn.init.orthogonal_(self.a.data)
        
    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        # Apply linear layer and sort nodes by head
        # Apply linear layer and sort nodes by head
        node_feats_i = self.projection1(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]
        node_feats_j = self.projection2(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]  

        # We need to calculate the attention logits for every edge in the adjacency matrix 
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat_i = node_feats_i.reshape(batch_size * num_nodes, self.num_heads, -1) # [4, 16]
        node_feats_flat_j = node_feats_j.reshape(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]



        # Calculate attention MLP output (independent for each head)
        attn_logits_l = torch.einsum('bhc,hc->bh', node_feats_flat_i[edge_indices_row], self.a_l) # [24, 4, 8] [4, 8] => [24, 4]
        attn_logits_r = torch.einsum('bhc,hc->bh', node_feats_flat_j[edge_indices_col], self.a_r) # [24, 4, 8] [4, 8] => [24, 4]
        attn_logits = self.leakyrelu(attn_logits_l + attn_logits_r)



        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15) # [2, 4, 4, 4]
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1) # [2, 4, 4, 4] B N_i N_j H
        
        # Weighted average of attention
        attn_probs = torch.softmax(attn_matrix, dim=2)
        # if print_attn_probs and log is not None:
        #     # print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        #     log.save_attention(attn_probs.permute(0, 3, 1, 2).detach().cpu().numpy())
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats_j) # [B N_i N_j H] [B N_j H C]
        


        # new 
        importance = torch.einsum('bjhc, hc->bjh', node_feats_j, self.a_r)
        importance_np = importance.permute(0, 2, 1).detach().cpu().numpy() # B H N_j
        # importance_np = np.repeat(importance_np[:, :, np.newaxis, :], importance.shape[2], 2) # [B, H, N_j, N_j]


        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats, importance_np # [B H N_j]



class GATLayer_exp_acc(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        self.projection1 = nn.Linear(c_in, c_out * num_heads)
        self.projection2 = nn.Linear(c_in, c_out * num_heads)
        self.a_l = nn.Parameter(torch.Tensor(num_heads, c_out))
        self.a_r = nn.Parameter(torch.Tensor(num_heads, c_out))
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.projection1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        
    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        node_feats_i = self.projection1(node_feats).view(batch_size, num_nodes, self.num_heads, -1)
        node_feats_j = self.projection2(node_feats).view(batch_size, num_nodes, self.num_heads, -1)

        # Calculate attention scores
        scores_l = torch.einsum('bijh,hc->bijh', node_feats_i, self.a_l)  # [B, N, H]
        scores_r = torch.einsum('bjkh,hc->bjkh', node_feats_j, self.a_r)  # [B, N, H]
        scores = self.leakyrelu(scores_l + scores_r.permute(0, 2, 1, 3))  # [B, N, N, H]

        # Weighted average of attention
        attn_probs = torch.softmax(scores, dim=2)
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats_j)

        # Compute importance
        importance = torch.einsum('bjhc,hc->bjh', node_feats_j, self.a_r)
        importance_np = importance.permute(0, 2, 1).detach().cpu().numpy()

        # Concatenate or average heads
        if self.concat_heads:
            node_feats = node_feats.view(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats, importance_np
    
    
class GATLayerV2_exp(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        """
        https://nn.labml.ai/graphs/gatv2/index.html
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection1 = nn.Linear(c_in, c_out * num_heads)
        self.projection2 = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # nn.init.orthogonal_(self.projection.weight.data)
        # nn.init.orthogonal_(self.a.data)
        
    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        node_feats_i = self.projection1(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]
        node_feats_j = self.projection2(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]  

        # We need to calculate the attention logits for every edge in the adjacency matrix 
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat_i = node_feats_i.reshape(batch_size * num_nodes, self.num_heads, -1) # [4, 16]
        node_feats_flat_j = node_feats_j.reshape(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]


        
        a_input = node_feats_flat_i[edge_indices_row] + node_feats_flat_j[edge_indices_col]
        
        a_input = self.leakyrelu(a_input)
        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a) # [24, 4, 8] [4, 8] => [24, 4]
        



        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15) # [2, 4, 4, 4]
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1) # [2, 4, 4, 4] B N_i N_j H
        
        # Weighted average of attention
        attn_probs = torch.softmax(attn_matrix, dim=2)
        # if print_attn_probs and log is not None:
        #     # print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        #     log.save_attention(attn_probs.permute(0, 3, 1, 2).detach().cpu().numpy())
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats_j) # [B N_i N_j H] [B N_j H C]
        

        # new 
        importance = torch.einsum('bjhc, hc->bjh', node_feats_j, self.a)
        importance_np = importance.permute(0, 2, 1).detach().cpu().numpy() # B H N_j
        # importance_np = np.repeat(importance_np[:, :, np.newaxis, :], importance.shape[2], 2) # [B, H, N_j, N_j]

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats, importance_np


class GATLayerV2_exp_acc(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, c_out))
        self.leakyrelu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
    def forward(self, node_feats, adj_matrix):
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and reshape for heads
        node_feats_proj = self.projection(node_feats).view(batch_size, num_nodes, self.num_heads, -1)

        # Calculate attention scores
        scores = torch.einsum('bijh,bkjh->bikh', node_feats_proj, node_feats_proj)  # [B, N, N, H]
        scores = self.leakyrelu(scores)

        # Weighted average of attention
        attn_probs = torch.softmax(scores, dim=2)
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats_proj)

        # Compute importance
        importance = torch.einsum('bjhc,hc->bjh', node_feats_proj, self.a)
        importance_np = importance.permute(0, 2, 1).detach().cpu().numpy()

        # Concatenate or average heads
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats, importance_np


class GATLayerV2_exp_pe(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        """
        https://nn.labml.ai/graphs/gatv2/index.html
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection1 = nn.Linear(c_in, c_out * num_heads)
        self.projection2 = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection1.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.projection2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # nn.init.orthogonal_(self.projection.weight.data)
        # nn.init.orthogonal_(self.a.data)
        
    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        node_feats_i = self.projection1(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]
        node_feats_j = self.projection2(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]  

        # We need to calculate the attention logits for every edge in the adjacency matrix 
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat_i = node_feats_i.reshape(batch_size * num_nodes, self.num_heads, -1) # [4, 16]
        node_feats_flat_j = node_feats_j.reshape(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        edge_indices_col = edges[:,0] * num_nodes + edges[:,2]


        
        a_input = node_feats_flat_i[edge_indices_row] + node_feats_flat_j[edge_indices_col]
        
        a_input = self.leakyrelu(a_input)
        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a) # [24, 4, 8] [4, 8] => [24, 4]
        



        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15) # [2, 4, 4, 4]
        attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1) # [2, 4, 4, 4] B N_i N_j H
        
        # Weighted average of attention
        attn_probs = torch.softmax(attn_matrix, dim=2)
        # if print_attn_probs and log is not None:
        #     # print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        #     log.save_attention(attn_probs.permute(0, 3, 1, 2).detach().cpu().numpy())
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats_j) # [B N_i N_j H] [B N_j H C]
        
        attn_matrix_np = attn_matrix.detach().cpu().numpy()
        attn_matrix_np_exp_win = attn_matrix_np[:, :-1, :-1, :].mean(axis=1) # 128, 16, 1
        win = attn_matrix_np[:, :-1, -1, :] # 128, 16, 1
        importance_np = attn_matrix_np_exp_win * win
        importance_np = importance_np.transpose(0, 2, 1)
        # importance_np = np.repeat(importance_np[:, :, np.newaxis, :], importance.shape[2], 2) # [B, H, N_j, N_j]

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats, importance_np


    
class GATLayerV2_exp_simplev1(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        """
        https://nn.labml.ai/graphs/gatv2/index.html
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection2 = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)


        nn.init.xavier_uniform_(self.projection2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # nn.init.orthogonal_(self.projection.weight.data)
        # nn.init.orthogonal_(self.a.data)
        
    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        
        node_feats_j = self.projection2(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]  

        
        # new 
        importance = torch.einsum('bjhc, hc->bjh', node_feats_j, self.a) # B N_j H
        attn_probs = torch.softmax(importance, dim=1)
        out = torch.einsum('bjh,bjhc->bjhc', attn_probs, node_feats_j).sum(dim=1).reshape(batch_size, -1) # [B, HC]

        importance_np = importance.permute(0, 2, 1).detach().cpu().numpy() # B H N_j
        # importance_np = np.repeat(importance_np[:, :, np.newaxis, :], importance.shape[2], 2) # [B, H, N_j, N_j]

        
        return out, importance_np

class GATLayerV2_exp_simplev2(nn.Module):
    
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=False, alpha=0.2):
        """
        https://nn.labml.ai/graphs/gatv2/index.html
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The 
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads
        
        # Sub-modules and parameters needed in the layer
        self.projection2 = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection2.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # nn.init.orthogonal_(self.projection.weight.data)
        # nn.init.orthogonal_(self.a.data)
        
    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, num_nodes, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        # Apply linear layer and sort nodes by head
        node_feats_j = self.projection2(node_feats).reshape(batch_size, num_nodes, self.num_heads, -1) # [1, 4, 16]  


        # new 
        importance = torch.einsum('bjhc, hc->bjh', node_feats_j, self.a)
        importance = importance.permute(0, 2, 1) # B H N_j
        importance = importance[:, :, None, :].repeat(1, 1, importance.shape[2], 1) # [B, H, N_j, N_j]


        # Weighted average of attention
        attn_probs = torch.softmax(importance, dim=2)
        # if print_attn_probs and log is not None:
        #     # print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        #     log.save_attention(attn_probs.permute(0, 3, 1, 2).detach().cpu().numpy())
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs.permute([0, 2, 3, 1]), node_feats_j) # [B N_i N_j H] [B N_j H C]


        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        
        return node_feats, importance.mean(dim=2).detach().cpu().numpy()






