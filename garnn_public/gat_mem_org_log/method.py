from .gat import *


class MLP_GATV2_exp_MLP(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))


        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayerV2_exp(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris=None, time_attris=None, attris_tar=None):
        if attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        else:
            X = G # for shap values
        B, T, C = X.shape

        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L


        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        if attris is None and X.shape[-1] > 1:
            return pred # for shap values
        
        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict
    
class MLP_GATV2_exp_MLP_acc(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))


        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayerV2_exp_acc(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris=None, time_attris=None, attris_tar=None):
        if attris is not None:
            X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        else:
            X = G # for shap values
        B, T, C = X.shape

        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L


        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        if attris is None and X.shape[-1] > 1:
            return pred # for shap values
        
        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict
    
    
    
from positional_encodings.torch_encodings import PositionalEncoding1D
class MLP_GATV2_exp_PE_MLP(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))


        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.pe = PositionalEncoding1D(1)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayerV2_exp(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        B, T, C = X.shape

        pe = self.pe(X[:, :, :1]).repeat(1, 1, C)       
       
        X = X + pe
        
        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L


        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict





from positional_encodings.torch_encodings import PositionalEncoding1D
class MLP_GATV2_exp_PE_with_Graph_MLP(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        num_varis = num_varis + 1
        
        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis
        
        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.pe = PositionalEncoding1D(1)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayerV2_exp_pe(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        

        pe = self.pe(X[:, :, :1])
       
        X = torch.concat([X, pe], dim=2)
        B, T, C = X.shape
        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C-1], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L


        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X[:, :, :-1]!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict
    
class MLP_GATV2_exp_PE_with_Graph_MLPv2(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        num_varis = num_varis + 1
        
        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis
        
        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.pe = PositionalEncoding1D(1)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayerV2_exp(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        

        pe = self.pe(X[:, :, :1])
       
        X = torch.concat([X, pe], dim=2)
        B, T, C = X.shape
        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L


        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict    
    
    
    
class MLP_GATsimplev2_exp_MLP(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))


        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayerV2_exp_simplev2(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        B, T, C = X.shape

        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L


        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict

class MLP_GAT_exp_MLP(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))


        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayer_exp(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        B, T, C = X.shape

        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L

        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict
    
class MLP_GAT_exp_MLP_acc(nn.Module):
    def __init__(self, hidden_dim, nhead, num_varis, num_layers, all_edges, alpha=0.2):
        super().__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))


        self.all_edges = all_edges
        self.feature_dim = hidden_dim // num_varis

        temp_list = [ nn.Sequential(
                init_(
                    nn.Linear(
                        1,
                        self.feature_dim,
                    )
                ),
                nn.ReLU(),
            ) for _ in range(num_varis)]
        self.encode_list = nn.ModuleList(temp_list)
        

        self.num_layers = num_layers

        self.nhead = nhead
        self.gat_layers = nn.ModuleList([GATLayer_exp_acc(self.feature_dim, self.feature_dim, num_heads=nhead, alpha=alpha) for _ in range(num_layers)])

        self.rnn = nn.GRUCell(self.feature_dim*num_varis, hidden_dim)

        temp_dim = hidden_dim
        self.output = nn.Sequential(
            init_(
                nn.Linear(
                    temp_dim,
                    temp_dim//2,
                )
            ),
            nn.ReLU(),
            init_(
                nn.Linear(
                    temp_dim//2,
                    1
                )
            ),
        )

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        B, T, C = X.shape

        X = X.reshape(B*T, C)

        X_emb_list = []
        for i in range(C):
            temp = self.encode_list[i](X[:, i].unsqueeze(dim=-1)) # B*T, C, F
            X_emb_list.append(temp)
        X_emb = torch.cat(X_emb_list, dim=1).reshape(B, T, C, self.feature_dim)
        
        X = X.reshape(B, T, C)

        part_msk = X.unsqueeze(dim=-1) # B, T, C, 1
        part_msk = torch.einsum('btcq,btqw->btcw',part_msk, part_msk.permute(0, 1, 3, 2)) # B T C C
        part_msk = (part_msk != 0)

        if self.all_edges:
            adj = torch.ones([B, T, C, C], dtype = X.dtype, device=X.device)
        else:
            adj = torch.tensor(part_msk.cpu().detach().numpy(), dtype = torch.float32, device = X.device)

        h = None
        all_attn_probs = np.zeros([B, T, self.nhead, self.num_layers, C], dtype=np.float32) # [B T H L N_j]
        
        for t in range(T):
        
            gnn_out = X_emb[:, t]
            for i in range(self.num_layers):
                gnn_out, attn_probs = self.gat_layers[i](gnn_out, adj[:, t])
                all_attn_probs[:, t, :, i, ...] = attn_probs
            h = self.rnn(gnn_out.view(B, -1), h) 

        all_attn_probs = all_attn_probs.mean(axis=2) # mean for H
        all_attn_probs = all_attn_probs.mean(axis=2) # mean for L

        vari_importance = np.mean(all_attn_probs, axis=1) # B C

        # B T C C
        # msk_np = msk.detach().cpu().numpy() if not self.all_edges else msk

        part_msk_np = (X!=0).detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)


        pred = self.output(h)

        output_dict = {
            'pred': pred,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_importance, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict

from .IMVLSTM import *
from .TGRU import *

class IMVTensorLSTM_NN(nn.Module):
    def __init__(self, hidden_dim, num_varis):
        super().__init__()
        self.imvlstm = IMVTensorLSTM(input_dim=num_varis, n_units=hidden_dim, output_dim=1)

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        mean, alphas, betas = self.imvlstm(X)
        output_dict = {
            'pred': mean,
            'all_attn_probs': alphas.squeeze(dim=-1).detach().cpu(), # B T C
            'variable_importance': betas.squeeze(dim=-1).detach().cpu(), # B C
            'part_variable_importance': betas.squeeze(dim=-1).detach().cpu() # B C
        }
        return output_dict

class IMVFullLSTM_NN(nn.Module):
    def __init__(self, hidden_dim, num_varis):
        super().__init__()
        self.imvlstm = IMVFullLSTM(input_dim=num_varis, n_units=hidden_dim, output_dim=1)

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        mean, alphas, betas = self.imvlstm(X)
        output_dict = {
            'pred': mean,
            'all_attn_probs': alphas.squeeze(dim=-1).detach().cpu(), # B T C
            'variable_importance': betas.squeeze(dim=-1).detach().cpu(), # B C
            'part_variable_importance': betas.squeeze(dim=-1).detach().cpu() # B C
        }
        return output_dict

class TGRU_NN(nn.Module):
    def __init__(self, hidden_dim, input_dim) :
        super().__init__()
        self.tgru = GRUModel(hidden_dim, input_dim)
        self.att = attention2_mix(hidden_dim, input_dim)

        
    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        h, _, _ = self.tgru(X)
        att_h_input = torch.stack(h, dim=1)
        qz_out, temp_weight, vari_weight = self.att(att_h_input)
        output_dict = {
            'pred': qz_out,
            'all_attn_probs': temp_weight, # B T C
            'variable_importance': vari_weight, # B C
            'part_variable_importance': vari_weight # B C
        }
        return output_dict

from .ode.odeint import *
class ETN_ODE_NN(nn.Module):
    def __init__(self, hidden_dim, input_dim, pred_window) :
        super().__init__()
        self.tgru = GRUModel(hidden_dim, input_dim)
        self.att = attention2_mix_org(hidden_dim, input_dim)
        self.func = ODEfunc(input_dim)
        self.pred_window = pred_window
        self.output = nn.Linear(input_dim, 1)
    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C
        h, _, _ = self.tgru(X)
        att_h_input = torch.stack(h, dim=1)
        qz_out, temp_weight, vari_weight = self.att(att_h_input)
        

        input_dim = X.shape[-1]
        qz0_mean, qz0_logvar = qz_out[:, :input_dim], qz_out[:, input_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(X.device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        total_T = torch.linspace(1, self.pred_window, self.pred_window).to(X.device)
        # forward in time and solve ode for reconstructions
        pred_z = odeint(self.func, z0, total_T).permute(1, 0, 2)
        
        pred = self.output(pred_z[:, -1, :])
        
        output_dict = {
            'pred': pred,
            'all_attn_probs': temp_weight, # B T C
            'variable_importance': vari_weight, # B C
            'part_variable_importance': vari_weight # B C
        }
        return output_dict

class ATT_F_LSTM_NN(nn.Module):
    def __init__(self, input_dim, time_length,  hidden_dim) :
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.f_att = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(time_length)])
        self.output = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim//2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim//2,
                1
            ),
        )
    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C

        temp_list = []
        att_list = []
        for t in range(X.shape[1]):
            feature_att = self.f_att[t](X[:,t,:])
            feature_att = torch.softmax(feature_att, dim=-1)
            new_x_t = X[:, t, :] * feature_att
            temp_list.append(new_x_t.unsqueeze(dim=1))
            att_list.append(feature_att.unsqueeze(dim=1))
        new_X = torch.cat(temp_list, dim=1)
        att = torch.cat(att_list, dim=1)
        hc = None
        for t in range(new_X.shape[1]):
            hc = self.lstm(new_X[:, t, :], hc)

        y = self.output(hc[0])

        part_att = att * (X!=0)

        output_dict = {
            'pred': y,
            'all_attn_probs': att.detach().cpu().numpy(), # B T C
            'variable_importance': att.mean(dim=1).detach().cpu().numpy(), # B C
            'part_variable_importance': part_att.mean(dim=1).detach().cpu().numpy() # B C
        }
        return output_dict
    
class ATT_T_LSTM_NN(nn.Module):
    def __init__(self, input_dim, time_length,  hidden_dim) :
        super().__init__()
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.t_att = nn.ModuleList([nn.Linear(time_length, time_length) for _ in range(input_dim)])
        self.output = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim//2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim//2,
                1
            ),
        )
    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B, T , C

        temp_list = []
        att_list = []
        for f in range(X.shape[-1]):
            temp_att = self.t_att[f](X[:,:,f])
            temp_att = torch.softmax(temp_att, dim=-1)
            new_x_t = X[:, :, f] * temp_att
            temp_list.append(new_x_t.unsqueeze(dim=-1))
            att_list.append(temp_att.unsqueeze(dim=-1))
        new_X = torch.cat(temp_list, dim=-1)
        att = torch.cat(att_list, dim=-1)
        hc = None
        for t in range(new_X.shape[1]):
            hc = self.lstm(new_X[:, t, :], hc)

        y = self.output(hc[0])

        part_att = att * (X!=0)
        
        output_dict = {
            'pred': y,
            'all_attn_probs': att.detach().cpu().numpy(), # B T C
            'variable_importance': att.mean(dim=1).detach().cpu().numpy(), # B C
            'part_variable_importance': part_att.mean(dim=1).detach().cpu().numpy() # B C
        }
        return output_dict



class RETAIN_NN(nn.Module):
    def __init__(self, hidden_dim, num_varis):
        super().__init__()
        self.gru_alpha = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.gru_beta = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.emb_layers = nn.Linear(num_varis, hidden_dim, bias=False)
        self.w_alpha = nn.Linear(hidden_dim, 1)
        self.w_beta = nn.Linear(hidden_dim, hidden_dim) # 
        self.output_layer = nn.Linear(hidden_dim, 1)
    

    def forward(self, G, attris, time_attris, attris_tar=None):
        X = torch.cat([torch.unsqueeze(G, dim=2), attris, time_attris], dim=2) # B T N

        emb_X = self.emb_layers(X) # B T C

        res_emb_X = torch.flip(emb_X, dims=(1,))
        self.gru_alpha.flatten_parameters()
        alpha, _ = self.gru_alpha(res_emb_X, None) # B T C

        alpha = self.w_alpha(alpha * 0.5) # B T 1

        alpha = torch.flip(alpha, dims=(1,))

        alpha = torch.softmax(alpha.squeeze(dim=-1), dim=-1) # B T


        self.gru_beta.flatten_parameters()
        beta, _ = self.gru_beta(res_emb_X, None) # B T C
        
        beta = self.w_beta(beta * 0.5) 

        beta = torch.flip(beta, dims=(1,))

        beta = torch.tanh(beta) # B T C

        c = (alpha.unsqueeze(dim=-1) * beta * emb_X).sum(dim=1) # B C

        y = self.output_layer(c) # B 1



        # [C N]
        beta_w = torch.einsum('cn, btc->btcn', self.emb_layers.weight, beta)  # B T C N
        # beta_w_temp = beta.unsqueeze(dim=-1) * self.emb_layers.weight.unsqueeze(dim=0).unsqueeze(dim=0)

        beta_w = beta_w.permute([0 ,1, 3, 2]) # B T N C
        # self.output_layer.weight 1 C
        all_attn_probs = torch.einsum('btnc, cq->btnq', beta_w, self.output_layer.weight.permute([1, 0]))
        
        all_attn_probs= all_attn_probs.squeeze(dim=-1) # B T N
        all_attn_probs = all_attn_probs * alpha.unsqueeze(dim=-1) * X
        all_attn_probs = all_attn_probs.detach().cpu().numpy()
        vari_import = np.mean(abs(all_attn_probs), axis=1) # Changed

        part_msk = (X != 0)
        part_msk_np = part_msk.detach().cpu().numpy()
        part_attn_probs = copy.deepcopy(all_attn_probs)
        part_attn_probs[~part_msk_np] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            part_variable_importance = np.nanmean(part_attn_probs, axis=1)

        output_dict = {
            'pred': y,
            'all_attn_probs': all_attn_probs, # B T C
            'variable_importance': vari_import, # B C
            'part_variable_importance': part_variable_importance # B C
        }

        return output_dict


from pytorch_forecasting.models.nhits.sub_modules import NHiTS as NHiTSModule
class NHiTSModule_NN(nn.Module):
    def __init__(self, time_length,  hidden_dim, output_hidden_dim) :
        super().__init__()
        # self.nhits = NHiTSModule(
        #     context_length=time_length,
        #     prediction_length=1,
        #     covariate_size=0,
        #     output_size=[output_hidden_dim],
        #     static_size=0,
        #     static_hidden_size=0,
        #     n_blocks=[1, 1, 1],
        #     n_layers=3 * [2], # n_blocks
        #     hidden_size=3 * [2 * [hidden_dim]], # n_blocks
        #     pooling_sizes=3 * [9],
        #     downsample_frequencies=3 * [1], # n_blocks 
        #     pooling_mode="average",
        #     interpolation_mode='linear',
        #     dropout=0.5,
        #     activation="ReLU",
        #     initialization="orthogonal",
        #     batch_normalization=True,
        #     shared_weights=True,
        #     naive_level=True,
        # )
        self.nhits = NHiTSModule(
            context_length=time_length,
            prediction_length=1,
            covariate_size=0,
            output_size=[output_hidden_dim],
            static_size=0,
            static_hidden_size=0,
            n_blocks=[1],
            n_layers=1 * [8], # n_blocks
            hidden_size=1 * [8 * [hidden_dim]], # n_blocks
            pooling_sizes=1 * [1],
            downsample_frequencies=1 * [1], # n_blocks 
            pooling_mode="max",
            interpolation_mode='nearest',
            dropout=0.0,
            activation="ReLU",
            initialization="orthogonal",
            batch_normalization=False,
            shared_weights=True,
            naive_level=True,
        )
        self.output = nn.Sequential(
            nn.Linear(
                output_hidden_dim,
                1,
            ),
        )
    def forward(self, G, attris, time_attris, attris_tar=None):

        X_tar = torch.unsqueeze(G, dim=2)
        X_fea = torch.cat([attris, time_attris], dim=2) # B, T , C   
        X_msk = torch.ones_like(X_tar[..., 0], device=X_fea.device)
        forecast, _, _, _ = self.nhits(X_tar, X_msk, X_fea, None, None)
        

        y = self.output(forecast.squeeze(dim=1))

                
        output_dict = {
            'pred': y,
            'all_attn_probs': None,
            'variable_importance': None,
            'part_variable_importance': None,
        }
        return output_dict
from pytorch_forecasting.models.nbeats.sub_modules import NBEATSGenericBlock
class NBEATSGenericBlock_NN(nn.Module):
    def __init__(self, time_length,  hidden_dim) :
        super().__init__()
        # self.nbeats = NBEATSGenericBlock(                    
        #     units=hidden_dim,
        #     thetas_dim=hidden_dim,
                        
        #     backcast_length=time_length,
        #     forecast_length=1,
        # )
        
        self.nbeats = NBEATSGenericBlock(                    
            units=hidden_dim,
            thetas_dim=hidden_dim,
            num_block_layers=1,
            dropout = 0.0,
            backcast_length=time_length,
            forecast_length=1,
        ) 

    def forward(self, G, attris, time_attris, attris_tar=None):

        X_tar = G
        backcast, forecast = self.nbeats(X_tar)
        

        y = forecast
                
        output_dict = {
            'pred': y,
            'all_attn_probs': None,
            'variable_importance': None,
            'part_variable_importance': None,
        }
        return output_dict
    
    
    
class LPSC_Cell(nn.Module):
    def __init__(self, closure_input_sizes,input_size=4,param_size=28,mlp_size=16,latent_size=5,num_hidden_layer=2,state_size=9):
        super(LPSC_Cell, self).__init__()
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(param_size)
        self.mlp2=MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.B=nn.Parameter(torch.zeros(input_size-2,latent_size))
        self.A=nn.Parameter(torch.zeros(latent_size,latent_size))
        self.switch=0
        mlp_struct2=[mlp_size]*num_hidden_layer
        mlp_struct2.append(1)
        # 和state size 一样多
        mlp_list=[MLP(closure_input_sizes[i],mlp_struct2,\
                      activation_layer=nn.ReLU, inplace=None, dropout=0) for i in range(state_size)]
        self.f = nn.ModuleList(mlp_list)
    def turn_on_closure(self):
        self.switch=1
        self.A.requires_grad=False
        self.B.requires_grad=False
        for (i,param) in enumerate(self.mlp2.parameters()):
            param.requires_grad = False
    def forward(self,inputs,hidden):
        #basal_insulin_data,bolus_c_data,bolus_t_data,\
        #intake_c_data,intake_t_data,\
        #hr_data, hr_d_data, time_start_data, time_end_data
        insulin=inputs[:,0:1]; carb=inputs[:,1:2]
        other=inputs[:,2:]
        
        #dynamical states
        S=hidden[1]
        Gp,Gt=torch.split(S[:,:2],1,dim=-1) #glucose states
        Ip,Il=torch.split(S[:,2:4],1,dim=-1) #insulin states
        X,XL=torch.split(S[:,4:6],1,dim=-1) #intermediary states
        Qsto1,Qsto2,Qgut=torch.split(S[:,6:],1,dim=-1) #meal absorption states
        #H,SRSH,Hsc1,Hsc2=torch.split(S[:,16:20],1,dim=-1) #glucagon states
        
        #parameters
        Z=hidden[0]
        new_Z=torch.matmul(Z,self.A)+torch.matmul(other,self.B)
        K=torch.abs(self.mlp2(new_Z))
        
        k1,k2=torch.split(K[:,0:2],1,dim=-1) #glucose params
        m1,m2,m3,m4=torch.split(K[:,2:6],1,dim=-1) #insulin params 
        kgri,D,kmin,kmax,kabs,alpha,beta,b,c,D,BW,f=torch.split(K[:,6:18],1,dim=-1) #ra params
        kp1,kp2,kp3,ki=torch.split(K[:,18:22],1,dim=-1) #EGP params
        Uii,Vm0,Vmx,Km0,r1,p2u=torch.split(K[:,22:],1,dim=-1) # U params
        #ke1,ke2=torch.split(K[:,35:37],1,dim=-1) #E params
        #ka1,ka2,kd=torch.split(K[:,35:39],1,dim=-1) #sub insulin/glucose params
        #n,delta,rho,sig1,sig2,srbh,kh1,kh2,kh3,SRBH=torch.split(K[:,41:51],1,dim=-1) #glucagon params
        
        #static states computation
        #EGP system
        EGP=kp1-kp2*Gp-kp3*XL#+xi*XH
        #Ra system
        Qsto=Qsto1+Qsto2        
        kemptQ=kmin+(kmax-kmin/2)*(\
               torch.tanh(alpha*(Qsto-b*D))-\
               torch.tanh(beta*(Qsto-c*D))+2)
        Ra=f*kabs*Qgut/BW
        
        #Utilization system
        Uid=(Vm0+Vmx*X*r1)*Gt/(Km0+torch.abs(Gt))
        
        #renal excretion system
        #E=ke1*torch.nn.functional.relu(Gp-ke2)
        
        #subcutaneous insulin kinetics
        #Rai=ka1*Isc1+ka2*Isc2
                
        #dynamic states update
        #glucagon states
        #DSRSH=-rho*(SRSH-torch.nn.functional.relu(sig*(Gth-G)+SRBH))
        #DHsc1=-(kh1+kh2)*Hsc1
        #DHsc2=kh1*Hsc1-kh3*Hsc2
        #DH=-n*H+SRH+RaH
        #DXH=-kH*XH+kH*torch.nn.functional.relu(H-Hb)
        
        #Meal
        c=self.switch
        DQsto1=-kgri*Qsto1+D*carb+c*self.f[0](torch.concat([Qsto1,carb],dim=-1))
        DQsto2=-kemptQ*Qsto2+kgri*Qsto1+c*self.f[1](torch.concat([Qsto1,Qsto2],dim=-1))
        DQgut=-kabs*Qgut+kemptQ*Qsto2+c*self.f[2](torch.concat([Qsto2,Qgut],dim=-1))
      
        #Utilization
        #insulin 
        DIp=-(m2+m4)*Ip+m1*Il+insulin+c*self.f[3](torch.concat([Ip,Il,insulin],dim=-1))
        DIl=-(m1+m3)*Il+m2*Ip+c*self.f[4](torch.concat([Il,Ip],dim=-1))
        DXL=-ki*(XL-Ip)+c*self.f[5](torch.concat([XL,Ip],dim=-1))
        DX=-p2u*X+p2u*Ip+c*self.f[6](torch.concat([X,Ip],dim=-1))
        #DIr=-ki*(Ir-I)
        #DIsc1=-(kd+ka1)*Isc1+insulin
        #DIsc2=kd*Isc1-ka2*Isc2
        
        #glucose
        DGp=EGP+Ra-Uii-k1*Gp+k2*Gt+c*self.f[7](torch.concat([XL,Qgut,Gp,Gt],dim=-1))
        DGt=-Uid+k1*Gp-k2*Gt+c*self.f[0](torch.concat([Gp,Gt],dim=-1))
        #DGs=-Ts*Gs+Ts*G
        
        new_S=torch.concat([Gp+DGp, Gt+DGt,\
                        Ip+DIp, Il+DIl,\
                        X+DX, XL+DXL,\
                        Qsto1+DQsto1, Qsto2+DQsto2, Qgut+DQgut],axis=-1)

        return Gp+DGp, [new_Z,new_S]
    
class LPSC_LSTM(nn.Module):
    def __init__(self,closre_input_sizes = [2,2,2,3,2,2,2,4,2], input_size=4, latent_size=16,state_size=9,\
                 output_size=1,mlp_size=32,num_hidden_layer=2):
        super(LPSC_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(state_size-1)
        self.mlp1 = MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.dtdcell=LPSC_Cell(closre_input_sizes,mlp_size=mlp_size,latent_size=latent_size,num_hidden_layer=num_hidden_layer,state_size=state_size)
        

    
    def turn_on_closure(self):
        for (i,param) in enumerate(self.lstm.parameters()):
            param.requires_grad = False
        for (i,param) in enumerate(self.mlp1.parameters()):
            param.requires_grad = False
        self.dtdcell.turn_on_closure()
    def forward(self, G, attris, time_attris, attris_tar=None):
    # def forward(self,past,s,x):
        
        past = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        s = past[:, -1, 0][..., None, None]
        past = past[:, :-1]
        x = attris_tar
        
        if x.shape[-1] < 4:
            pad = torch.zeros([x.shape[0], x.shape[1], 4 - x.shape[-1]], device=G.device)
            
            x = torch.cat([x, pad], dim=-1)
            
            pad = torch.zeros([past.shape[0], past.shape[1], 5 - past.shape[-1]], device=G.device)
            
            past = torch.cat([past, pad], dim=-1)
            
        
        #past is N*L*5
        self.lstm.flatten_parameters() 
        lstm_out, (h0,c0)=self.lstm(past)
        hidden=[c0[0],torch.concat([s[:,0],self.mlp1(h0[0])],axis=-1)]
        pred, hidden = self.dtdcell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dtdcell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
            
            
        output_dict = {
            'pred': pred[:, -1,:],
            'all_attn_probs': None,
            'variable_importance': None,
            'part_variable_importance': None,
        }
        return output_dict



class DTDSimCell(nn.Module):
    def __init__(self,  input_size=4, param_size=28, mlp_size=16, latent_size=5, num_hidden_layer=2):
        super(DTDSimCell, self).__init__()
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(param_size)
        self.mlp2=MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.B=nn.Parameter(torch.zeros(input_size-2,latent_size))
        self.A=nn.Parameter(torch.zeros(latent_size,latent_size))
    def forward(self,inputs,hidden):
        #basal_insulin_data,bolus_c_data,bolus_t_data,\
        #intake_c_data,intake_t_data,\
        #hr_data, hr_d_data, time_start_data, time_end_data
        insulin=inputs[:,0:1]; carb=inputs[:,1:2]
        other=inputs[:,2:]
        
        #dynamical states
        S=hidden[1]
        Gp,Gt=torch.split(S[:,:2],1,dim=-1) #glucose states
        Ip,Il=torch.split(S[:,2:4],1,dim=-1) #insulin states
        X,XL=torch.split(S[:,4:6],1,dim=-1) #intermediary states
        Qsto1,Qsto2,Qgut=torch.split(S[:,6:],1,dim=-1) #meal absorption states
        #H,SRSH,Hsc1,Hsc2=torch.split(S[:,16:20],1,dim=-1) #glucagon states
        
        #parameters
        Z=hidden[0]
        new_Z=torch.matmul(Z,self.A)+torch.matmul(other,self.B)
        K=torch.abs(self.mlp2(new_Z))
        
        
        k1,k2=torch.split(K[:,0:2],1,dim=-1) #glucose params
        m1,m2,m3,m4=torch.split(K[:,2:6],1,dim=-1) #insulin params 
        kgri,D,kmin,kmax,kabs,alpha,beta,b,c,D,BW,f=torch.split(K[:,6:18],1,dim=-1) #ra params
        kp1,kp2,kp3,ki=torch.split(K[:,18:22],1,dim=-1) #EGP params
        Uii,Vm0,Vmx,Km0,r1,p2u=torch.split(K[:,22:],1,dim=-1) # U params
        #ke1,ke2=torch.split(K[:,35:37],1,dim=-1) #E params
        #ka1,ka2,kd=torch.split(K[:,35:39],1,dim=-1) #sub insulin/glucose params
        #n,delta,rho,sig1,sig2,srbh,kh1,kh2,kh3,SRBH=torch.split(K[:,41:51],1,dim=-1) #glucagon params
        
        #static states computation
        #EGP system
        EGP=kp1-kp2*Gp-kp3*XL#+xi*XH
        #Ra system
        Qsto=Qsto1+Qsto2        
        kemptQ=kmin+(kmax-kmin/2)*(\
               torch.tanh(alpha*(Qsto-b*D))-\
               torch.tanh(beta*(Qsto-c*D))+2)
        Ra=f*kabs*Qgut/BW
        
        #Utilization system
        Uid=(Vm0+Vmx*X*r1)*Gt/(Km0+torch.abs(Gt))
        
        #renal excretion system
        #E=ke1*torch.nn.functional.relu(Gp-ke2)
        
        #subcutaneous insulin kinetics
        #Rai=ka1*Isc1+ka2*Isc2
                
        #dynamic states update
        #glucagon states
        #DSRSH=-rho*(SRSH-torch.nn.functional.relu(sig*(Gth-G)+SRBH))
        #DHsc1=-(kh1+kh2)*Hsc1
        #DHsc2=kh1*Hsc1-kh3*Hsc2
        #DH=-n*H+SRH+RaH
        #DXH=-kH*XH+kH*torch.nn.functional.relu(H-Hb)
        
        #Meal
        DQsto1=-kgri*Qsto1+D*carb
        DQsto2=-kemptQ*Qsto2+kgri*Qsto1
        DQgut=-kabs*Qgut+kemptQ*Qsto2
      
        #Utilization
        #insulin 
        DIp=-(m2+m4)*Ip+m1*Il+insulin
        DIl=-(m1+m3)*Il+m2*Ip
        DXL=-ki*(XL-Ip)
        DX=-p2u*X+p2u*Ip
        #DIr=-ki*(Ir-I)
        #DIsc1=-(kd+ka1)*Isc1+insulin
        #DIsc2=kd*Isc1-ka2*Isc2
        
        #glucose
        DGp=EGP+Ra-Uii-k1*Gp+k2*Gt
        DGt=-Uid+k1*Gp-k2*Gt
        #DGs=-Ts*Gs+Ts*G
        
        new_S=torch.concat([Gp+DGp, Gt+DGt,\
                        Ip+DIp, Il+DIl,\
                        X+DX, XL+DXL,\
                        Qsto1+DQsto1, Qsto2+DQsto2, Qgut+DQgut],axis=-1)

        return Gp+DGp, [new_Z,new_S]
    
class DTD_LSTM(nn.Module):
    def __init__(self,input_size=4,latent_size=16,state_size=9,\
                 output_size=1,mlp_size=32,num_hidden_layer=2):
        super(DTD_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        mlp_struct=[mlp_size]*num_hidden_layer
        mlp_struct.append(state_size-1)
        self.mlp1 = MLP(latent_size,mlp_struct,\
                     activation_layer=nn.ReLU, inplace=None,dropout=0)
        self.dtdcell=DTDSimCell(mlp_size=mlp_size,latent_size=latent_size,num_hidden_layer=num_hidden_layer)
    def forward(self, G, attris, time_attris, attris_tar=None):
    # def forward(self,past,s,x):
        
        past = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        s = past[:, -1, 0][..., None, None]
        past = past[:, :-1]
        x = attris_tar
        
        if x.shape[-1] < 4:
            pad = torch.zeros([x.shape[0], x.shape[1], 4 - x.shape[-1]], device=G.device)
            
            x = torch.cat([x, pad], dim=-1)
            
            pad = torch.zeros([past.shape[0], past.shape[1], 5 - past.shape[-1]], device=G.device)
            
            past = torch.cat([past, pad], dim=-1)
            
        #past is N*L*5
        self.lstm.flatten_parameters() 
        lstm_out, (h0,c0)=self.lstm(past)
        hidden=[c0[0],torch.concat([s[:,0],self.mlp1(h0[0])],axis=-1)]
        pred, hidden = self.dtdcell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dtdcell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
            pred = torch.concat([pred,new_pred],axis=1)
        output_dict = {
            'pred': pred[:, -1,:],
            'all_attn_probs': None,
            'variable_importance': None,
            'part_variable_importance': None,
        }
        return output_dict

class DAG_RNN(nn.Module):
    def __init__(self, DAG, input_size, output_ind, mlp_size,num_hidden_layers,activation,dropout):
        super(DAG_RNN, self).__init__()
        self.dag=DAG
        self.state_size = len(DAG)
        self.output_ind = output_ind
        self.mlp_inputs = [DAG[i][-1][0] for i in range(self.state_size)]
        mlp_struct=[mlp_size]*num_hidden_layers
        mlp_struct.append(1)
        mlp_list=[MLP(self.mlp_inputs[i],mlp_struct,\
                      activation_layer=activation, inplace=None, dropout=dropout) for i in range(self.state_size)]
        self.f = nn.ModuleList(mlp_list)
        #self.h = MLP(self.state_size,[mlp_size,mlp_size,output_size])
    def forward(self, input, hidden):

        d_hidden=[]
        for i in range(len(self.f)):
            mlp_in=torch.concat([hidden[:,self.dag[i][0]],input[:,self.dag[i][1]]],axis=-1)
            d_hidden.append(self.f[i](mlp_in))
        new_hidden=hidden+torch.concat(d_hidden,axis=-1)
        return new_hidden[:,self.output_ind:self.output_ind+1], new_hidden 
    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.state_size)
    
    
class MNODE_LSTM(nn.Module):
    def __init__(
        self,
        DAG=[[[0,3,4,5,6],[],[5]],\
                            [[1],[0],[2]],\
                            [[1,2],[],[2]],\
                            [[2,3],[],[2]],\
                            [[4],[2,3],[3]],\
                            [[4,5],[],[2]],\
                            [[6],[1],[2]]],

                 input_size=4,latent_size=7,output_ind=0,\
                 mlp_size=32,num_hidden_layers=2,activation=nn.ReLU,dropout=0):
        super(MNODE_LSTM, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size+1,\
                  hidden_size=latent_size,\
                  num_layers=2,\
                  batch_first=True)
        self.dag_rnn_cell=DAG_RNN(DAG,input_size,output_ind,mlp_size,num_hidden_layers,\
                              activation=activation,dropout=dropout)
    def forward(self, G, attris, time_attris, attris_tar=None):
    # def forward(self,past,s,x):
        
        past = torch.cat([torch.unsqueeze(G, dim=2), attris], dim=2) # B, T , C
        s = past[:, -1, 0][..., None, None]
        past = past[:, :-1]
        x = attris_tar
        
        if x.shape[-1] < 4:
            pad = torch.zeros([x.shape[0], x.shape[1], 4 - x.shape[-1]], device=G.device)
            
            x = torch.cat([x, pad], dim=-1)
            
            pad = torch.zeros([past.shape[0], past.shape[1], 5 - past.shape[-1]], device=G.device)
            
            past = torch.cat([past, pad], dim=-1)
        #past is N*L*5
        self.lstm.flatten_parameters() 
        lstm_out, (h0,_)=self.lstm(past)
        h0=h0[0]
        h0=h0[:,1:]
        hidden=torch.concat([s[:,0],h0],axis=-1)
        pred, hidden = self.dag_rnn_cell(x[:,0],hidden)
        pred = torch.unsqueeze(pred,axis=1)
        for j in range(1,x.shape[1]):
            new_pred, hidden = self.dag_rnn_cell(x[:,j],hidden)
            new_pred = torch.unsqueeze(new_pred,axis=1)
        output_dict = {
            'pred': pred[:, -1,:],
            'all_attn_probs': None,
            'variable_importance': None,
            'part_variable_importance': None,
        }
        return output_dict
    