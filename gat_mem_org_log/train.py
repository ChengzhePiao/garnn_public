from .method import *
from .update import *
from .dataprocessor import *
from .log import *

def get_model(CONF):
    
    if CONF['model_name'] == 'gatv2_exp':
        return MLP_GATV2_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    
    elif CONF['model_name'] == 'gatv2_exp_acc':
        return MLP_GATV2_exp_MLP_acc(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    
    
    elif  CONF['model_name'] == 'gatv2_exp_trick':
        return MLP_GATV2_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gat_exp':
        return MLP_GAT_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gat_exp_acc':
        return MLP_GAT_exp_MLP_acc(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gat_exp_trick':
        return MLP_GAT_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gatv2_exp_layer_1':
        return MLP_GATV2_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=1,
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gatv2_exp_layer_1_acc':
        return MLP_GATV2_exp_MLP_acc(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=1,
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    
    elif CONF['model_name'] == 'gat_exp_layer_1':
        return MLP_GAT_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=1,
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gat_exp_layer_1_acc':
        return MLP_GAT_exp_MLP_acc(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=1,
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    
    elif CONF['model_name'] == 'imv_tensor_lstm':
        return IMVTensorLSTM_NN(hidden_dim=CONF['hidden_dim'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        )
    elif CONF['model_name'] == 'imv_full_lstm':
        return IMVFullLSTM_NN(hidden_dim=CONF['hidden_dim'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        )
    elif CONF['model_name'] == 'tgru':
        return TGRU_NN(hidden_dim=CONF['hidden_dim'], 
        input_dim=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        )
    elif CONF['model_name'] == 'ETN-ODE':
        return ETN_ODE_NN(hidden_dim=CONF['hidden_dim'], 
        input_dim=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        pred_window = CONF['pred_window']
        )
    elif CONF['model_name'] == 'att_f_lstm':
        return ATT_F_LSTM_NN(
            input_dim=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
            time_length= CONF['seq_len'],
            hidden_dim=CONF['hidden_dim'],
        )
    elif CONF['model_name'] == 'att_t_lstm':
        return ATT_T_LSTM_NN(
            input_dim=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
            time_length= CONF['seq_len'],
            hidden_dim=CONF['hidden_dim'],
        )
    elif CONF['model_name'] == 'retain':
        return RETAIN_NN(hidden_dim=CONF['hidden_dim'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        )
    
    elif CONF['model_name'] == 'gat_simplev2':
        return MLP_GATsimplev2_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=CONF['num_layers'],
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gat_simplev2_layer_1':
        return MLP_GATsimplev2_exp_MLP(hidden_dim=CONF['hidden_dim'], 
        nhead=CONF['nhead'], 
        num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
        num_layers=1,
        all_edges=CONF['all_edges'],
        alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gatv2_exp_pe':
        return MLP_GATV2_exp_PE_MLP(
            hidden_dim=CONF['hidden_dim'], 
            nhead=CONF['nhead'], 
            num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
            num_layers=CONF['num_layers'],
            all_edges=CONF['all_edges'],
            alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gatv2_exp_pe_graph':
        return MLP_GATV2_exp_PE_with_Graph_MLP(
            hidden_dim=CONF['hidden_dim'], 
            nhead=CONF['nhead'], 
            num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
            num_layers=CONF['num_layers'],
            all_edges=CONF['all_edges'],
            alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'gatv2_exp_pe_graphv2':
        return MLP_GATV2_exp_PE_with_Graph_MLPv2(
            hidden_dim=CONF['hidden_dim'], 
            nhead=CONF['nhead'], 
            num_varis=len(CONF['attri_list']) + len(CONF['time_attri_list']) + 1, 
            num_layers=CONF['num_layers'],
            all_edges=CONF['all_edges'],
            alpha=CONF['alpha'] if 'alpha'in CONF else 0.2
        )
    elif CONF['model_name'] == 'NHiTS':
        return NHiTSModule_NN(
            time_length=CONF['seq_len'],
            hidden_dim=CONF['hidden_dim'],
            output_hidden_dim=CONF['hidden_dim']//8
        )
        
    elif CONF['model_name'] == 'NBEATS':
        return NBEATSGenericBlock_NN(
            time_length=CONF['seq_len'],
            hidden_dim=CONF['hidden_dim'],
        )
        
    elif CONF['model_name'] == 'LPSC':
        return LPSC_LSTM()
    
    elif CONF['model_name'] == 'LP':
        return DTD_LSTM()

    elif CONF['model_name'] == 'MNODE':
        return MNODE_LSTM()

def create_pid2model(data_proc, CONF, model):
    pid2model = {}
    for pid in data_proc.test_pid2data_npy:
        pid2model[pid] = get_model(CONF)
        pid2model[pid].to(CONF['device'])
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        pid2model[pid].load_state_dict(copy.deepcopy(model.state_dict())) 
        # same with pid2model[pid].load_state_dict(model.state_dict()) no difference
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return pid2model

def log_epoch_variable_importance(whether_log, epoch_index, update, models, data_proc, log):
    if not whether_log:
        return
    part_variable_importance_list = []
    variable_importance_list = []

    for pid in data_proc.train_pid2data_npy.keys():
        if type(models) == dict:
            model = models[pid]
        else:
            model = models
        output_dict = update.test_all_train_inference(model, pid)
        part_variable_importance_list.append(output_dict['part_variable_importance'])
        variable_importance_list.append(output_dict['variable_importance'])
    if output_dict['variable_importance'] is not None:
        variable_importance = np.concatenate(variable_importance_list, axis=0)
        part_variable_importance = np.concatenate(part_variable_importance_list, axis=0)
        log.record_summary_vari_importance_for_epochs(variable_importance, part_variable_importance, epoch_index=epoch_index)


def update_metrics_dic(metrics_dic, per_metrics_dic):
    for mtc in per_metrics_dic:
        if mtc not in metrics_dic:
            metrics_dic[mtc] = []
        metrics_dic[mtc].append(per_metrics_dic[mtc])
    return metrics_dic

def print_metrics_dic(metrics_dic):

    for mtc in metrics_dic:
        print(f'{mtc}, {np.mean(metrics_dic[mtc]):.5f}, {np.std(metrics_dic[mtc]):.5f}')
    
    print(metrics_dic['rmse'])


def eval_inference(data_proc, inference, best_pid2model, log, comments, print_metrics):
    pid2prediction = {}
    part_variable_importance_list = []
    variable_importance_list = []
    metrics_dic = {}
    for pid in data_proc.train_pid2data_npy.keys():
        # if 'save_attention' not in CONF or not CONF['save_attention']:
        # update.test_inference
        output_dict = inference(best_pid2model[pid], pid)

        prediction, variable_importance, part_variable_importance = output_dict['pred'], output_dict['variable_importance'], output_dict['part_variable_importance']
        if 'per_metrics_dic' in output_dict:
            update_metrics_dic(per_metrics_dic=output_dict['per_metrics_dic'], metrics_dic=metrics_dic)
        if variable_importance is not None:
            variable_importance_list.append(variable_importance)
        if part_variable_importance is not None:    
            part_variable_importance_list.append(part_variable_importance)
        pid2prediction[pid] = prediction
    if variable_importance is not None:
        variable_importance = np.concatenate(variable_importance_list, axis=0)
        part_variable_importance = np.concatenate(part_variable_importance_list, axis=0)
        log.summary_vari_importance(variable_importance, name=f'{comments}_all_variable_importance')
        log.summary_vari_importance(part_variable_importance, name=f'{comments}_all_variable_importance_trick')
    # if temp_attn_probs is not None:
    #     temp_attn_probs = np.concatenate(temp_attn_probs_list, axis=0)
    #     log.save_temp_att_importance(temp_attn_probs)
    # log.save_vari_temp_importance(all_attn_probs, 10, temp_attn_probs)

    if print_metrics:
        print_metrics_dic(metrics_dic)
        log.save_prediction(pid2prediction)
        log.save_metrics_dic(metrics_dic)


def train(CONF):
    global_model = get_model(CONF)
    global_model.to(CONF['device'])
    global_model.train()

    best_val_rmse = np.inf
    best_model = None

    data_proc = DataProcessor(CONF)

    log = Log(CONF['dataset'] + '_' + CONF['model_name'] + '_' + CONF['comments'], CONF)

    if 'rewrite' in CONF and not CONF['rewrite'] and os.path.exists(os.path.join(log.root_dir, 'pid2prediction.npy')):
        print('exists.... don\'t rewrite')
        return
    
    update = Update(data_proc, CONF)
    best_val_rmse = np.inf

    pbar = tqdm(range(CONF['epochs']), desc = 'global_loop')


    log_epoch_variable_importance(CONF['log_epoch_vari_imp'], 0, update, global_model, data_proc, log)
    # log.record_eval_rmse(np.mean(eval_rmse_list))

    for epoch in pbar:

        update.update_weights(
                model=global_model, local_epochs=CONF['local_epochs'], lr=CONF['lr'])


        if epoch % CONF['print_every'] == 0:

            rmse_list = []
            mape_list = []
            for pid in data_proc.valid_pid2data_npy.keys():
                output_dict = update.inference(global_model, pid)
                rmse, mape = output_dict['rmse'], output_dict['mape']
                rmse_list.append(rmse)
                mape_list.append(mape)
            # log.record_summary_vari_importance_for_epochs(all_attn_probs, epoch_index=(epoch + 1) * CONF['local_epochs'])
            log_epoch_variable_importance(CONF['log_epoch_vari_imp'], (epoch + 1) * CONF['local_epochs'], update, global_model, data_proc, log)

            log.record_eval_rmse(np.mean(rmse_list))
            rmse = np.mean(rmse_list)
            mape = np.mean(mape_list)

            if best_val_rmse > rmse:
                best_val_rmse = rmse
                best_model = copy.deepcopy(global_model)

            best_model = copy.deepcopy(global_model)



    metrics_dic = {}
    for pid in data_proc.test_pid2data_npy.keys():

        output_dict = update.test_inference(best_model, pid)
        per_metrics_dic = output_dict['per_metrics_dic']
        metrics_dic = update_metrics_dic(metrics_dic=metrics_dic, per_metrics_dic=per_metrics_dic)

    print(' \n Results after ' +
        str(CONF['epochs'])+' global rounds of training:')
    print_metrics_dic(metrics_dic)

    tune_pbar = tqdm(range(CONF['tuning_epochs']), desc = 'tune_loop')
    pid2model = create_pid2model(data_proc, CONF, best_model)
    best_pid2model = copy.deepcopy(pid2model)
    
    if CONF['model_name'] == 'LPSC':
        for pid in best_pid2model:
            best_pid2model[pid].turn_on_closure()
            pid2model[pid].turn_on_closure()
    
    
    pid2best_val_rmse = {}

    # best_val_rmse = np.inf
    for epoch in tune_pbar:
        eval_rmse_list = []
        for pid in pid2model:
            update.update_weights(
                model=pid2model[pid], local_epochs=CONF['tuning_local_epochs'], pid = pid, lr=CONF['tuning_lr'])

            
            if epoch % CONF['print_every'] == 0:                  

                output_dict = update.inference(pid2model[pid], pid)
                rmse = output_dict['rmse']
                eval_rmse_list.append(rmse)
                
                if pid not in pid2best_val_rmse:
                    pid2best_val_rmse[pid] = np.inf
                if pid2best_val_rmse[pid] > rmse:
                    pid2best_val_rmse[pid] = rmse
                    best_pid2model[pid] = copy.deepcopy(pid2model[pid])

        if epoch % CONF['print_every'] == 0:
            log_epoch_variable_importance(CONF['log_epoch_vari_imp'], CONF['epochs'] * CONF['local_epochs'] + (epoch + 1) * CONF['tuning_local_epochs'],
                                        update, pid2model, data_proc, log)
            log.record_eval_rmse(np.mean(eval_rmse_list))
    
    if CONF['model_name'] == 'LPSC':
        for pid in best_pid2model:
            best_pid2model[pid].turn_on_closure()
            pid2model[pid].turn_on_closure()
    
    if CONF['log_epoch_vari_imp']:
        log.save_epoch_vari_importance(is_part=True)
        log.save_epoch_vari_importance(is_part=False)
        log.save_epoch_rmse()


    print(' \n Results after ' +
        str(CONF['tuning_epochs'])+' of fine tuning:')
    
    eval_inference(data_proc=data_proc, inference=update.test_inference, best_pid2model=best_pid2model, log=log, comments='test', print_metrics=True)

    eval_inference(data_proc=data_proc, inference=update.test_all_train_inference, best_pid2model=best_pid2model, log=log, comments='train', print_metrics=False)
    
    if CONF['save_model']:
        log.save_models(best_pid2model)
        print('models have been saved...')
    

    return best_pid2model
