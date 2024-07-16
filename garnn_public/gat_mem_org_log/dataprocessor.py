from util import *

class DataProcessor():
    def __init__(self, CONF):
        self.CONF = CONF
        n_prev = CONF['n_prev']
        pred_window = CONF['pred_window']
        self.train_pid2data_npy = np.load(
            os.path.join(CONF['data_path'], f'train_pid2data_npy_{n_prev}_{pred_window}.npy'), 
            allow_pickle=True
        )[()]
        self.valid_pid2data_npy = np.load(
            os.path.join(CONF['data_path'], f'valid_pid2data_npy_{n_prev}_{pred_window}.npy'), 
            allow_pickle=True
        )[()]
        self.test_pid2data_npy = np.load(
            os.path.join(CONF['data_path'], f'test_pid2data_npy_{n_prev}_{pred_window}.npy'),  
            allow_pickle=True
        )[()]

        self.all_train_data = {}
        for pid in self.train_pid2data_npy:
            for content in self.train_pid2data_npy[pid]:
                if content not in self.all_train_data:
                    self.all_train_data[content] = self.train_pid2data_npy[pid][content]
                elif content == 'mean' or content == 'std':
                    self.all_train_data[content] += self.train_pid2data_npy[pid][content]
                else:

                    self.all_train_data[content] = np.concatenate([self.all_train_data[content],
                    self.train_pid2data_npy[pid][content]], axis=0) 
        self.all_train_data['mean'] /= len(self.train_pid2data_npy)
        self.all_train_data['std'] /= len(self.train_pid2data_npy)

        self.attri2idx = pd.read_pickle(os.path.join(CONF['data_path'], 'attri2idx.pkl'))
        

    def get_train_batch(self, batch_size, device, seq_len, attri_list, time_attri_list, pid=None, ):

        data = self.all_train_data if pid is None else self.train_pid2data_npy[pid]
        
        idxs = np.random.choice(data['y'].shape[0], batch_size)

        G = data[f'glucose_level_X'][idxs, -seq_len:]
        
        temp_data = data['attri_X'][idxs, -seq_len:, :] # select batch
        attris = temp_data[:, :, self.attri2idx.loc[attri_list, 'idx'].to_numpy()] # select features
        
        temp_data = data['attri_X_tar'][idxs]
        attris_tar = temp_data[:, :, self.attri2idx.loc[attri_list, 'idx'].to_numpy()]
        
        time_attris = temp_data[:, :, self.attri2idx.loc[time_attri_list, 'idx'].to_numpy()] # select features

        y = data['y'][idxs]

        G = torch.tensor(G, dtype=torch.float32, device=device)
        attris = torch.tensor(attris, dtype=torch.float32, device=device)
        attris_tar = torch.tensor(attris_tar, dtype=torch.float32, device=device)
        time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)
        y = torch.tensor(np.expand_dims(y, axis=1), dtype=torch.float32, device=device)

        target_time_list = []
        for time_attri in time_attri_list:
            time_name = f'target_{time_attri}'
            time_np = np.expand_dims(data[time_name][idxs], axis=1)
            target_time_list.append(time_np)
        if len(target_time_list)!=0:
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        return G, attris, time_attris, y, target_time, attris_tar
        

    def get_val(self, device, seq_len, attri_list, time_attri_list, pid):
        data = self.valid_pid2data_npy[pid]
        mean = self.valid_pid2data_npy[pid]['mean']
        std = self.valid_pid2data_npy[pid]['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        attris_tar = data['attri_X_tar'][:, :, self.attri2idx.loc[attri_list, 'idx']]
        time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        attris = torch.tensor(attris, dtype=torch.float32, device=device)
        attris_tar = torch.tensor(attris_tar, dtype=torch.float32, device=device)
        time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        target_time_list = []
        for time_attri in time_attri_list:
            time_name = f'target_{time_attri}'
            time_np = np.expand_dims(data[time_name], axis=1)
            target_time_list.append(time_np)
        if len(target_time_list)!=0:
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        return G, attris, time_attris, y, target_time, attris_tar

    def get_mean_std(self, pid):
        return self.test_pid2data_npy[pid]['mean'],  self.test_pid2data_npy[pid]['std']

    def get_test(self, device, seq_len, attri_list, time_attri_list, pid):
        data = self.test_pid2data_npy[pid]
        mean = self.test_pid2data_npy[pid]['mean']
        std = self.test_pid2data_npy[pid]['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        attris_tar = data['attri_X_tar'][:, :, self.attri2idx.loc[attri_list, 'idx']]
        time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        attris = torch.tensor(attris, dtype=torch.float32, device=device)
        attris_tar = torch.tensor(attris_tar, dtype=torch.float32, device=device)
        time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        target_time_list = []
        for time_attri in time_attri_list:
            time_name = f'target_{time_attri}'
            time_np = np.expand_dims(data[time_name], axis=1)
            target_time_list.append(time_np)
        if len(target_time_list)!=0:
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        seq_st_ed = data['seq_st_ed_list']

        return G, attris, time_attris, y, target_time, seq_st_ed, attris_tar
    
    def get_all_train(self, device, seq_len, attri_list, time_attri_list, pid=None):
        if pid is not None:
            data = self.train_pid2data_npy[pid]
            mean = self.train_pid2data_npy[pid]['mean']
            std = self.train_pid2data_npy[pid]['std']
        else:
            data = self.all_train_data
            mean = self.all_train_data['mean']
            std = self.all_train_data['std']
        
        G = data[f'glucose_level_X'][:, -seq_len:]
        attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[attri_list, 'idx']]
        attris_tar = data['attri_X_tar'][:, :, self.attri2idx.loc[attri_list, 'idx']]
        time_attris = data['attri_X'][:, -seq_len:, self.attri2idx.loc[time_attri_list, 'idx']]
        y = data['y'] * std + mean

        G = torch.tensor(G, dtype=torch.float32, device=device)
        attris = torch.tensor(attris, dtype=torch.float32, device=device)
        attris_tar = torch.tensor(attris_tar, dtype=torch.float32, device=device)
        time_attris = torch.tensor(time_attris, dtype=torch.float32, device=device)

        target_time_list = []
        for time_attri in time_attri_list:
            time_name = f'target_{time_attri}'
            time_np = np.expand_dims(data[time_name], axis=1)
            target_time_list.append(time_np)
        if len(target_time_list)!=0:
            target_time = torch.tensor(np.concatenate(target_time_list, axis=1), dtype=torch.float32, device=device)
        else:
            target_time = None

        return G, attris, time_attris, y, target_time, attris_tar