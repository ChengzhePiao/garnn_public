from util import *
import math

class GRUCell(nn.Module):
    def __init__(self, num_units, n_var):
        super(GRUCell, self).__init__()

        self._num_units = num_units
        self._linear = None
        self._input_dim = None

        # added by mv_rnn
        self._n_var = n_var
        self.gate_w_I2H = nn.Parameter(
            torch.zeros([n_var, 1, 2*int(num_units / n_var), 1],dtype=torch.float32),
            requires_grad=True)
        self.gate_w_H2H = nn.Parameter(
            torch.zeros([n_var, 1, 2*int(num_units / n_var), int(num_units / n_var)],dtype=torch.float32),
            requires_grad=True)
        self.gate_bias = nn.Parameter(
            torch.ones(2*num_units,dtype=torch.float32),
            requires_grad=True)
        self.update_w_I2H = nn.Parameter(
            torch.zeros([n_var, 1, int(num_units / n_var), 1], dtype=torch.float32),
            requires_grad=True)
        self.update_w_H2H = nn.Parameter(
            torch.zeros([n_var, 1, int(num_units / n_var), int(num_units / n_var)], dtype=torch.float32),
            requires_grad=True)
        self.update_bias = nn.Parameter(
            torch.ones(num_units, dtype=torch.float32),
            requires_grad=True)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self._num_units)
        for w in self.parameters():
            w.data.uniform_(-std, std)


    def forward(self, inputs, hidden):
        # reshape input
        # [B H]
        blk_input = inputs.unsqueeze(2).permute(1, 0, 2)
        mv_input = blk_input.unsqueeze(2)
        # reshape hidden
        B = hidden.size()[0]
        blk_h = torch.unsqueeze(hidden, dim=2)
        blk_h2 = blk_h.view(B, -1, self._n_var)
        mv_h = blk_h2.permute(2, 0, 1).unsqueeze(2)

        gate_tmp_I2H = (mv_input * self.gate_w_I2H).sum(3)
        gate_tmp_H2H = (mv_h * self.gate_w_H2H).sum(3)
        # [V 1 B 2D]
        gate_tmp_x = torch.chunk(gate_tmp_I2H, self._n_var, dim=0)
        gate_tmp_h = torch.chunk(gate_tmp_H2H, self._n_var, dim=0)
        # [1 B 2H]
        g_res_x = torch.cat(gate_tmp_x, dim=2)
        g_res_h = torch.cat(gate_tmp_h, dim=2)

        gate_res_x = g_res_x.squeeze(0)
        gate_res_h = g_res_h.squeeze(0)
        # [B 2H;
        res_gate = gate_res_x + gate_res_h + self.gate_bias
        z, r = torch.chunk(res_gate, 2, dim=1)

        blk_r = torch.unsqueeze(torch.sigmoid(r), dim=2)
        blk_r2 = blk_r.view(B, -1, self._n_var)
        mv_r = blk_r2.permute(2, 0, 1).unsqueeze(2)

        update_tmp_I2H = (mv_input * self.update_w_I2H).sum(3)
        update_tmp_H2H = ((mv_h * mv_r) * self.update_w_H2H).sum(3)

        update_tmp_x = torch.chunk(update_tmp_I2H, self._n_var, dim=0)
        update_tmp_h = torch.chunk(update_tmp_H2H, self._n_var, dim=0)

        u_res_x = torch.cat(update_tmp_x, dim=2)
        u_res_h = torch.cat(update_tmp_h, dim=2)

        update_res_x = u_res_x.squeeze(0)
        update_res_h = u_res_h.squeeze(0)

        g = update_res_x + update_res_h + self.update_bias
        new_h = torch.sigmoid(z) * hidden + (1 - torch.sigmoid(z)) * torch.tanh(g)


        return new_h


class GRUModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, ):
        super(GRUModel,self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(hidden_dim, input_dim)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim * 2)
        #self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)

        outs = []

        hn = h0
        # reverse time step for input series
        for seq in reversed(range(x.size(1))):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)

        h_first = outs[-1].squeeze()
        qz_out = self.fc_q(h_first)
        # h_last = outs[-1].squeeze()
        #out = self.fc(h_last)
        return outs, h_first, qz_out


class attention2_mix(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(attention2_mix, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        per_vari_dim = int(hidden_dim / input_dim)
        self.per_vari_dim = per_vari_dim
        self.att_w_temp = nn.Parameter(torch.zeros([input_dim, 1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = nn.Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.att_w_vari = nn.Parameter(torch.zeros([per_vari_dim, 1], dtype=torch.float32), requires_grad=True)
        self.bias_vari = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = nn.Parameter(torch.zeros([1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.att_w_mul = nn.Parameter(torch.zeros([1, input_dim, 1], dtype=torch.float32), requires_grad=True)
        self.fc = nn.Linear(input_dim, 1)
        #self.fc = nn.Linear(input_dim, 20)
        #self.fc_mu = nn.Linear(input_dim,input_dim)
        #self.fc_var = nn.Linear(input_dim,input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, att_inputs ):
        #att_h_tensor = torch.stack(att_inputs, dim=1)
        att_h_tensor = att_inputs
        att_h_list = torch.split(att_h_tensor, [self.per_vari_dim] * self.input_dim, 2)

        h_att_temp_input = att_h_list
        # temporal attention
        # [V B T D]
        tmph = torch.stack(h_att_temp_input, dim=0)
        # [V B T-1 D], [V, B, 1, D]
        #tmph_before, tmph_last = torch.split(tmph, [self.steps - 1, 1], 2)
        # -- temporal logits

        temp_logit = torch.tanh((tmph * self.att_w_temp).sum(3) + self.bias_temp)


        temp_weight = torch.softmax(temp_logit, dim=-1)
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        tmph_cxt = (tmph * temp_weight.unsqueeze(-1)).sum(2)
        #tmph_last = tmph_last.squeeze(2)
        v_temp = (tmph * temp_weight.unsqueeze(-1)).sum(3).permute(1, 0, 2)

        # [V B 2D]
        #h_temp = torch.cat((tmph_last, tmph_cxt), 2)
        h_temp = tmph_cxt

        # variable attention

        vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
        # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True)).permute(1, 0, 2))
        vari_weight = torch.softmax(vari_logits, dim=1)
        # [B V D]
        h_trans = h_temp.permute(1, 0, 2)
        # [B D]
        h_weighted = (h_trans * vari_weight).sum(1)


        c_t = (v_temp * vari_weight).sum(2)
        qz_out = self.fc(c_t)
        #ct_mean = (v_temp * vari_weight).mean(2)
        #mu = self.fc_mu(ct_mean)
        #var = self.fc_var(c_t)

        #return c_t, qz_out
        #return mu, var
        return qz_out, temp_weight.permute([1, 2, 0]).detach().cpu().numpy()[:, ::-1, :], vari_weight.squeeze(dim=-1).detach().cpu().numpy()


class attention2_mix_org(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(attention2_mix_org, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        per_vari_dim = int(hidden_dim / input_dim)
        self.per_vari_dim = per_vari_dim
        self.att_w_temp = nn.Parameter(torch.zeros([input_dim, 1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.bias_temp = nn.Parameter(torch.ones([input_dim, 1, 1], dtype=torch.float32), requires_grad=True)
        self.att_w_vari = nn.Parameter(torch.zeros([per_vari_dim, 1], dtype=torch.float32), requires_grad=True)
        self.bias_vari = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.aug_w_vari = nn.Parameter(torch.zeros([1, 1, per_vari_dim], dtype=torch.float32), requires_grad=True)
        self.att_w_mul = nn.Parameter(torch.zeros([1, input_dim, 1], dtype=torch.float32), requires_grad=True)
        self.fc = nn.Linear(input_dim, input_dim * 2)
        #self.fc = nn.Linear(input_dim, 20)
        #self.fc_mu = nn.Linear(input_dim,input_dim)
        #self.fc_var = nn.Linear(input_dim,input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, att_inputs ):
        #att_h_tensor = torch.stack(att_inputs, dim=1)
        att_h_tensor = att_inputs
        att_h_list = torch.split(att_h_tensor, [self.per_vari_dim] * self.input_dim, 2)

        h_att_temp_input = att_h_list
        # temporal attention
        # [V B T D]
        tmph = torch.stack(h_att_temp_input, dim=0)
        # [V B T-1 D], [V, B, 1, D]
        #tmph_before, tmph_last = torch.split(tmph, [self.steps - 1, 1], 2)
        # -- temporal logits

        temp_logit = torch.tanh((tmph * self.att_w_temp).sum(3) + self.bias_temp)


        temp_weight = torch.softmax(temp_logit, dim=-1)
        # temp_before [V B T-1 D], temp_weight [V B T-1]
        tmph_cxt = (tmph * temp_weight.unsqueeze(-1)).sum(2)
        #tmph_last = tmph_last.squeeze(2)
        v_temp = (tmph * temp_weight.unsqueeze(-1)).sum(3).permute(1, 0, 2)

        # [V B 2D]
        #h_temp = torch.cat((tmph_last, tmph_cxt), 2)
        h_temp = tmph_cxt

        # variable attention

        vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True) + self.bias_vari).permute(1, 0, 2))
        # vari_logits = torch.tanh(((h_temp * self.aug_w_vari).sum(2, keepdim=True)).permute(1, 0, 2))
        vari_weight = torch.softmax(vari_logits, dim=1)
        # [B V D]
        h_trans = h_temp.permute(1, 0, 2)
        # [B D]
        h_weighted = (h_trans * vari_weight).sum(1)


        c_t = (v_temp * vari_weight).sum(2)
        qz_out = self.fc(c_t)
        #ct_mean = (v_temp * vari_weight).mean(2)
        #mu = self.fc_mu(ct_mean)
        #var = self.fc_var(c_t)

        #return c_t, qz_out
        #return mu, var
        return qz_out, temp_weight.permute([1, 2, 0]).detach().cpu().numpy()[:, ::-1, :], vari_weight.squeeze(dim=-1).detach().cpu().numpy()



class ODEfunc(nn.Module):
    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.nfe = 0

    def forward(self, t, h):
        self.nfe += 1
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh