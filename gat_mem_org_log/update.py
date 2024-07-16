from statistics import mode
from attr import attr
from sklearn.metrics import mean_absolute_error
from util import *

def MAPE_LOSS(pred, y):
    return torch.mean(
            torch.abs((y-pred))/(y+1e-6)
        )
class Update():
    def __init__(self, data_proc, CONF):

        self.data_proc = data_proc
        self.criterion = nn.MSELoss()
        self.CONF = CONF

    def update_weights(self, model, local_epochs, pid=None, lr = None):

        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=self.CONF['weight_decay'])

        for iter in range(local_epochs):
            batch_loss = []


            G, attris, time_attris, y, _, attris_tar = self.data_proc.get_train_batch(
            self.CONF['batch_size'], self.CONF['device'], 
            self.CONF['seq_len'], self.CONF['attri_list'], self.CONF['time_attri_list'], pid=pid)

            model.zero_grad()

            output_dict = model(G, attris, time_attris, attris_tar)

            predicted_norm = output_dict['pred']

            loss = self.criterion(predicted_norm, y) # + 0.001*MAPE_LOSS(predicted_norm, y)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        
        epoch_loss = np.mean(batch_loss)
        
        
        # if self.CONF['model_name'] == 'LPSC':
        #     model.turn_on_closure()
            
        return model, epoch_loss

    def inference(self, model, pid):
        
        mean, std = self.data_proc.get_mean_std(pid)
        model.eval()
        with torch.no_grad():

            G, attris, time_attris, y, _, attris_tar = self.data_proc.get_val(self.CONF['device'], self.CONF['seq_len'], self.CONF['attri_list'], self.CONF['time_attri_list'], pid)
            
            
            temp_batch = 128
            st = 0
            predicted_norm_list, variable_importance_list, part_variable_importance_list = [], [], []
            while True:
                ed = st + temp_batch
                # print(st, ed, G[st:ed].shape[0], G.shape[0], )
                output_dict = model(G[st:ed], attris[st:ed], time_attris[st:ed], attris_tar[st:ed])
                predicted_norm_, all_attn_probs_, variable_importance_, part_variable_importance_ = output_dict['pred'], output_dict['all_attn_probs'], output_dict['variable_importance'], output_dict['part_variable_importance']
                predicted_norm_list.append(predicted_norm_)
                if variable_importance_ is not None:
                    variable_importance_list.append(variable_importance_)
                if part_variable_importance_ is not None:
                    part_variable_importance_list.append(part_variable_importance_)
                st = ed
                if ed >= G.shape[0]:
                    break
            predicted_norm = torch.cat(predicted_norm_list, dim=0)

            variable_importance = np.concatenate(variable_importance_list, axis=0) if len(variable_importance_list) > 0 else None
            part_variable_importance = np.concatenate(part_variable_importance_list, axis=0) if len(part_variable_importance_list) > 0 else None

            if predicted_norm.shape[0] != G.shape[0]:
                print('error')



            predicted_norm_np = predicted_norm.cpu().numpy()[:]

            predicted_np = predicted_norm_np * std + mean
            rmse = mean_squared_error(y, predicted_np[:,0])**0.5
            mape = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100


            output_dict = {
                'rmse': rmse,
                'mape': mape,
                'variable_importance': variable_importance,
                'part_variable_importance' : part_variable_importance,
            }
        return output_dict

    def test_inference(self, model, pid, log=None, print_feature_maps=False):
        mean, std = self.data_proc.get_mean_std(pid)
        model.eval()
        with torch.no_grad():
   
            G, attris, time_attris, y, _, seq_st_ed, attris_tar = self.data_proc.get_test(self.CONF['device'], self.CONF['seq_len'],  self.CONF['attri_list'], self.CONF['time_attri_list'], pid)

                        # model.to(self.CONF['device'])    
            temp_batch = 128
            st = 0
            predicted_norm_list, variable_importance_list, part_variable_importance_list = [], [], []
            all_attn_probs_list = [] if print_feature_maps else None
            while True:
                ed = st + temp_batch
                # print(st, ed, G[st:ed].shape[0], G.shape[0], )
                output_dict = model(G[st:ed], attris[st:ed], time_attris[st:ed], attris_tar[st:ed])
                predicted_norm_, all_attn_probs_, variable_importance_, part_variable_importance_ = output_dict['pred'], output_dict['all_attn_probs'], output_dict['variable_importance'], output_dict['part_variable_importance']
                predicted_norm_list.append(predicted_norm_)

                if all_attn_probs_list is not None and all_attn_probs_ is not None:
                    all_attn_probs_list.append(all_attn_probs_)              
                if variable_importance_ is not None:
                    variable_importance_list.append(variable_importance_)
                if part_variable_importance_ is not None:
                    part_variable_importance_list.append(part_variable_importance_)
                st = ed
                if ed >= G.shape[0]:
                    break
            predicted_norm = torch.cat(predicted_norm_list, dim=0)

            variable_importance = np.concatenate(variable_importance_list, axis=0) if len(variable_importance_list) > 0 else None
            part_variable_importance = np.concatenate(part_variable_importance_list, axis=0) if len(part_variable_importance_list) > 0 else None
            all_attn_probs = np.concatenate(all_attn_probs_list, axis=0) if all_attn_probs_list is not None and len(all_attn_probs_list) > 0 else None

            if predicted_norm.shape[0] != G.shape[0]:
                print('error')

            predicted_norm_np = predicted_norm.cpu().numpy()[:]
            predicted_np = predicted_norm_np * std + mean
        
        output_dict = {}
        per_metrics_dic ={}
        per_metrics_dic['rmse'] = mean_squared_error(y, predicted_np[:,0])**0.5
        per_metrics_dic['mape'] = mean_absolute_percentage_error(y, predicted_np[:,0]) * 100
        per_metrics_dic['mae'] = mean_absolute_error(y, predicted_np[:,0])
        per_metrics_dic['grmse'] = cal_gmse(y, predicted_np)
        per_metrics_dic['time_lag'] = cal_time_lag(y, predicted_np, seq_st_ed, self.CONF['interval'])
        output_dict['per_metrics_dic'] = per_metrics_dic
        output_dict['pred'] = predicted_np
        output_dict['all_attn_probs'] = all_attn_probs
        output_dict['variable_importance'] = variable_importance
        output_dict['part_variable_importance'] = part_variable_importance


        return output_dict
    
    def test_all_train_inference(self, model, pid, log=None, print_feature_maps=False):
        mean, std = self.data_proc.get_mean_std(pid)
        model.eval()
        with torch.no_grad():
   
            G, attris, time_attris, y, _, attris_tar = self.data_proc.get_all_train(self.CONF['device'], self.CONF['seq_len'],  self.CONF['attri_list'], self.CONF['time_attri_list'], pid)

            # model.to(self.CONF['device'])    
            temp_batch = 128
            st = 0
            predicted_norm_list, variable_importance_list, part_variable_importance_list = [], [], []
            all_attn_probs_list = [] if print_feature_maps else None
            while True:
                ed = st + temp_batch
                # print(st, ed, G[st:ed].shape[0], G.shape[0], )
                output_dict = model(G[st:ed], attris[st:ed], time_attris[st:ed], attris_tar[st:ed])
                predicted_norm_, all_attn_probs_, variable_importance_, part_variable_importance_ = output_dict['pred'], output_dict['all_attn_probs'], output_dict['variable_importance'], output_dict['part_variable_importance']
                predicted_norm_list.append(predicted_norm_)

                if all_attn_probs_list is not None and all_attn_probs_ is not None:
                    all_attn_probs_list.append(all_attn_probs_)              
                if variable_importance_ is not None:
                    variable_importance_list.append(variable_importance_)
                if part_variable_importance_ is not None:
                    part_variable_importance_list.append(part_variable_importance_)
                st = ed
                if ed >= G.shape[0]:
                    break
            predicted_norm = torch.cat(predicted_norm_list, dim=0)

            variable_importance = np.concatenate(variable_importance_list, axis=0) if len(variable_importance_list) > 0 else None
            part_variable_importance = np.concatenate(part_variable_importance_list, axis=0) if len(part_variable_importance_list) > 0 else None
            all_attn_probs = np.concatenate(all_attn_probs_list, axis=0) if all_attn_probs_list is not None and len(all_attn_probs_list) > 0 else None

            if predicted_norm.shape[0] != G.shape[0]:
                print('error')
            # model.to(self.CONF['device'])
            predicted_norm_np = predicted_norm.cpu().numpy()[:]
            predicted_np = predicted_norm_np * std + mean

        output_dict = {}
        output_dict['pred'] = predicted_np
        output_dict['all_attn_probs'] = all_attn_probs
        output_dict['variable_importance'] = variable_importance
        output_dict['part_variable_importance'] = part_variable_importance
        

        return output_dict