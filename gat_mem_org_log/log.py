from util import *

class Log():
    def __init__(self, name, CONF):
        self.root_dir = os.path.join(CONF['log_path'], name)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.global_epoch = []
        self.tune_epoch = []
        self.CONF = CONF

        self.epoch_vari_importance = []
        self.epoch_part_vari_importance = []
        self.epoch_indexs = []
    
        self.eval_rmse_list = []
        
    def save_prediction(self, pid2prediction, ):
        
        save_path = os.path.join(self.root_dir, 'pid2prediction.npy')
        np.save(save_path, pid2prediction)

    def save_metrics_dic(self, metrics_dic):
        save_path = os.path.join(self.root_dir, 'pid2metrics.npy')
        np.save(save_path, metrics_dic)

    def save_models(self, pid2model):
        for pid in pid2model:
            save_path = os.path.join(self.root_dir, f'{pid}_model.pth')
            torch.save(pid2model[pid].state_dict(), save_path)
    def save_scikit_models(self, pid2model):
        for pid in pid2model:
            save_path = os.path.join(self.root_dir, f'{pid}_model.joblib')
            dump(pid2model[pid], save_path) 
    def load_scikit_models(self, pid2model):
        path = self.root_dir[:-14]
        for pid in pid2model:
            load_path = os.path.join(path, f'{pid}_model.joblib')
            pid2model[pid] = load(load_path) 
        return pid2model
    
    def save_xgboost_models(self, pid2model):
        for pid in pid2model:
            save_path = os.path.join(self.root_dir, f'{pid}_model.json')
            pid2model[pid].save_model(save_path)
    def load_xgboost_models(self, pid2model):
        path = self.root_dir[:-14]
        for pid in pid2model:
            load_path = os.path.join(path, f'{pid}_model.json')
            pid2model[pid].load_model(load_path) 
        return pid2model
    
    def load_models(self, pid2model, CONF, pid=None):


        if pid is not None:
            path = self.root_dir
            load_path = os.path.join(path, f'{pid}_model.pth')
            pid2model.load_state_dict(torch.load(load_path))
            pid2model.eval()
            return pid2model
        for pid in pid2model:
            try:
                path = self.root_dir[:-14]
                load_path = os.path.join(path, f'{pid}_model.pth')
                pid2model[pid].load_state_dict(torch.load(load_path))
            except:
                path = self.root_dir
                load_path = os.path.join(path, f'{pid}_model.pth')
                pid2model[pid].load_state_dict(torch.load(load_path))
            
            pid2model[pid].eval()
        return pid2model
    def save_rmse(self, pid2rmse):
        save_path = os.path.join(self.root_dir, 'pid2rmse.npy')
        np.save(save_path, pid2rmse)
    
    def save_mape(self, pid2mape):
        save_path = os.path.join(self.root_dir, 'pid2mape.npy')
        np.save(save_path, pid2mape)

    def save_mae(self, pid2mae):
        save_path = os.path.join(self.root_dir, 'pid2mae.npy')
        np.save(save_path, pid2mae)

    def save_attention(self, attention):
        save_path = os.path.join(self.root_dir, f'{self.current_pid}_{self.t}_attention.npy')
        np.save(save_path, attention)

    def set_pid(self, pid):
        self.current_pid = pid

    def set_time(self, t):
        self.t = t

    def save_global_epochs(self, epoch, rmse, time):
        save_path = os.path.join(self.root_dir, 'global_epoch.npy')

        self.global_epoch.append([epoch, rmse, time])

        np.save(save_path, np.array(self.global_epoch))

    def save_tune_epochs(self, epoch, rmse, time):
        save_path = os.path.join(self.root_dir, 'tune_epoch.npy')

        self.tune_epoch.append([epoch, rmse, time])

        np.save(save_path, np.array(self.tune_epoch))


    def record_eval_rmse(self, rmse):
        self.eval_rmse_list.append(rmse)

    def save_epoch_rmse(self):
        save_path = os.path.join(self.root_dir, 'epoch_rmse.pdf')
        fig, ax = plt.subplots(figsize = (20, 10))
        ax.plot(self.epoch_indexs[1:],self.eval_rmse_list, )
        ax.set_ylabel('RMSE (mg/dL)', fontsize=24)
        ax.set_xlabel('Epoch', fontsize=24)   # relative to plt.rcParams['font.size']
        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        fig.tight_layout()
        plt.savefig(save_path)
        plt.clf()


    def summary_vari_importance(self, variable_importance, gen_figure=True, name=None):

        # all_attn_probs [B C]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if len(variable_importance.shape) == 2:
                final_importance = np.nanmean(variable_importance, axis = 0)
            else:
                final_importance = variable_importance

        sort_idxs = np.argsort(-final_importance)

        attris = ['glucose_level'] + self.CONF['attri_list'] + self.CONF['time_attri_list']

        if gen_figure:
            fig, ax = plt.subplots(figsize = (20, 15))
            bars = ax.barh(np.arange(len(attris)) , [round(d, 3) for d in final_importance[sort_idxs]], 0.5)
            # bars = ax.barh(np.arange(len(attris)) , final_importance[sort_idxs], 0.5)
            for bars in ax.containers:
                ax.bar_label(bars, size=34)
            ax.set_xlabel('Variable importance', fontsize=34)
            ax.set_ylabel('Variable', fontsize=34)   # relative to plt.rcParams['font.size']
            new_attris = [attris[idx] for idx in sort_idxs]
            ax.set_yticks(np.arange(len(new_attris)), new_attris)
            ax.xaxis.set_tick_params(labelsize=34)
            ax.yaxis.set_tick_params(labelsize=34)
            
            save_path = os.path.join(self.root_dir, 'all_variable_importance.pdf') if name is None else os.path.join(self.root_dir, f'{name}.pdf')
            fig.tight_layout()
            plt.savefig(save_path)
            plt.clf()
            np.save(os.path.join(self.root_dir, f'{name}.npy'), [new_attris, final_importance[sort_idxs]])
        
        return final_importance


    def record_summary_vari_importance_for_epochs(self, varible_importance, part_variable_importance, epoch_index):
        final_variable_importance = self.summary_vari_importance(varible_importance, gen_figure=False)
        final_part_variable_importance = self.summary_vari_importance(part_variable_importance, gen_figure=False)
        self.epoch_vari_importance.append(np.expand_dims(final_variable_importance, axis=0))
        self.epoch_part_vari_importance.append(np.expand_dims(final_part_variable_importance, axis=0))
        self.epoch_indexs.append(epoch_index)

    def save_epoch_vari_importance(self, is_part=False):
        from matplotlib.cm import get_cmap

        name = "tab20"
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        colors = cmap.colors  # type: list
        name_pre = '_part' if is_part else ''
        save_path = os.path.join(self.root_dir, f'epoch{name_pre}_variable_importance.pdf')
        epoch_vari_importance = np.concatenate(self.epoch_vari_importance, axis=0) if not is_part else np.concatenate(self.epoch_part_vari_importance, axis=0)
        fig, ax = plt.subplots(figsize = (30, 10))
        ax.set_prop_cycle(color=colors)
        attris = ['glucose_level'] + self.CONF['attri_list'] + self.CONF['time_attri_list']
        attris2idx_val = {}
        for i in range(epoch_vari_importance.shape[1]):
            ax.plot(self.epoch_indexs, epoch_vari_importance[:, i], label=attris[i], linewidth=3)
            attris2idx_val[attris[i]] = (self.epoch_indexs, epoch_vari_importance[:, i])
        ax.set_ylabel('Variable importance', fontsize=34)
        ax.set_xlabel('Epoch', fontsize=34)   # relative to plt.rcParams['font.size']
        ax.xaxis.set_tick_params(labelsize=34)
        ax.yaxis.set_tick_params(labelsize=34)
        ax.legend(fontsize=34, ncol=3, loc='best')
        fig.tight_layout()
        np.save(os.path.join(self.root_dir, f'epoch{name_pre}_variable_importance.npy'), attris2idx_val)
        plt.savefig(save_path)
        plt.clf()
        

    # def save_temp_att_importance(self, temp_attn_probs):
    #     save_path = os.path.join(self.root_dir, 'temporal_importance.pdf')
    #     mean_val = np.mean(temp_attn_probs, axis=0)
    #     std_val = np.mean(temp_attn_probs, axis=0)
    #     plt.rcParams['figure.figsize'] = (20, 15)
    #     plt.plot(np.arange(temp_attn_probs.shape[1]), mean_val, marker='o')
    #     for i in np.arange(temp_attn_probs.shape[1]):
    #         plt.text(i, mean_val[i], f'{mean_val[i]:.2f}' + r'$\pm$' + f'{std_val[i]:.2f}', ha='center', va='bottom', fontsize=10, )
    #     plt.xlabel('t', fontsize=34)
    #     plt.ylabel('Temporal importance', fontsize=24)
    #     plt.xticks(fontsize=34)
    #     plt.yticks(fontsize=34)
    #     plt.tight_layout()
    #     plt.savefig(save_path)
    #     plt.clf()

    def save_vari_temp_importance(self, all_attn_probs, num_samples=None, temp_attn_probs=None):
        if num_samples is not None:
            idxs = np.random.choice(all_attn_probs.shape[0], size=num_samples, replace=False)
            all_attn_probs = all_attn_probs[idxs]
            if temp_attn_probs is not None:
                temp_attn_probs = temp_attn_probs[idxs]
        print(all_attn_probs.shape)
        all_attn_probs = np.nanmean(all_attn_probs, axis=2).transpose([0, 2, 1]) # B C T
        all_attn_probs = np.nan_to_num(all_attn_probs)
        print(all_attn_probs.shape)
        B, C, T = all_attn_probs.shape
        attris = ['glucose_level'] + self.CONF['attri_list'] + self.CONF['time_attri_list']
        for i in range(all_attn_probs.shape[0]):
            if temp_attn_probs is None:
                nrows = 1
                sharex = False
            else:
                nrows = 2
                sharex = True
            fig, axs = plt.subplots(figsize = (20, 10), ncols=1, nrows=nrows, sharex = sharex)

            ax = axs if nrows == 1 else axs[0]
            im = ax.imshow(all_attn_probs[i]) 

            ax.set_xticks(np.arange(T), labels=np.arange(T), fontsize=24)
            ax.set_yticks(np.arange(C), labels=attris, fontsize=24)
            ax.set_xlabel('t', fontsize=24)
            ax.set_ylabel('Variable', fontsize=24)   # relative to plt.rcParams['font.size']
            
            if temp_attn_probs is None:
                # Create colorbar
                cbar = ax.figure.colorbar(im, ax=ax, orientation='horizontal')
                # cbar = ax.figure.colorbar(im, ax=ax,)
                cbar.ax.set_ylabel('Variable importance', rotation=-90, va="bottom", fontsize=24)
                cbar.ax.yaxis.set_tick_params(labelsize=24)


            valfmt = '{x:.2f}'
            # Get the formatter in case a string is supplied
            if isinstance(valfmt, str):
                valfmt = mpl.ticker.StrMethodFormatter(valfmt)

            # # Loop over data dimensions and create text annotations.
            for c in range(C):
                for t in range(T):
                    text = ax.text(t, c, valfmt(all_attn_probs[i, c, t], None),
                                ha="center", va="center", fontsize=7)
            # print(all_attn_probs[i].min(), all_attn_probs[i].max())

            save_path = os.path.join(self.root_dir, f'vari_temp_importance_sample_{i}.pdf')

            if temp_attn_probs is None:
                fig.tight_layout()
                
                plt.savefig(save_path)
                plt.clf()
            if temp_attn_probs is not None:
                temp = np.ones([C,T], dtype=np.float32) * np.expand_dims(temp_attn_probs[i], axis=0)
                
                ax = axs[1]
                im = ax.imshow(temp) 

                ax.set_xticks(np.arange(T), labels=np.arange(T), fontsize=24)
                ax.set_yticks(np.arange(C), labels=attris, fontsize=24)
                ax.set_xlabel('t', fontsize=24)
                ax.set_ylabel('Variable', fontsize=24)   # relative to plt.rcParams['font.size']
                
                # Create colorbar
                
                cbar = fig.colorbar(im, ax=axs.ravel().tolist())
                cbar.ax.set_ylabel('Variable importance', rotation=-90, va="bottom", fontsize=25)
                cbar.ax.yaxis.set_tick_params(labelsize=25)


                for j in np.arange(temp_attn_probs.shape[1]):
                    ax.text(j, C//2, f'{temp_attn_probs[i, j]:.2f}', ha='center', va='bottom', fontsize=25, )
                # fig.subplots_adjust(right=0.6, left=0.1, top=1.8, bottom=0.6, wspace=1.9)
                fig.tight_layout()
                plt.savefig(save_path, bbox_inches='tight')
                plt.clf()
