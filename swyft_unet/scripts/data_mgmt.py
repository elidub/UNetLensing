import os
import numpy as np

def print_existance(file, path):
    print(f'{file} {path} exists!') if os.path.exists(path) else print(f'{file} does not exist!')
    
    store_entry = {'m': m, 'nsub': nsub, 'nsim': nsim, 'system_name': system_name}
    
def get_ids(entry):
              
    store_dict = {}
    for e in ['m', 'nsub', 'nsim']:
        if e in entry: store_dict[e] = entry[e]
        
    dataset_dict = store_dict.copy()
    if 'sigma' in entry: dataset_dict['sigma'] = entry['sigma']
    
    mre_dict = dataset_dict.copy()
    if 'nmc' in entry: mre_dict['nmc'] = entry['nmc']
    
    pred_dict = mre_dict.copy()
    if 'npred' in entry: pred_dict['npred'] = entry['npred']
    
    dicts = [store_dict, dataset_dict, mre_dict, pred_dict]
    ids = []
    for d in dicts:
        id_string = '_'.join(np.array(list(d.items())).flatten())
        if 'simul' in entry: id_string = f"{entry['simul']}_" + id_string
        id_string = '_' + id_string
        ids.append(id_string)
    
    return ids

def get_store_path(store_id, store_dir = '/nfs/scratch/eliasd/store'):
    store_name = f'store{store_id}.zarr'
    store_path = os.path.join(store_dir, store_name)
    return store_path
                     
def get_dataset_path(dataset_id, store_dir = '/nfs/scratch/eliasd/dataset'):
    dataset_name = f'dataset{dataset_id}.pt'
    dataset_path = os.path.join(store_dir, dataset_name)
    return dataset_path

def get_mre_path(mre_id, store_dir = '../data/mre'):
    mre_name = f'mre{mre_id}.pt'
    mre_path = os.path.join(store_dir, mre_name)
    return mre_path

def get_pred_path(pred_id, store_dir = '../data/pred'):
    pred_name = f'pred{pred_id}.pickle'
    pred_path = os.path.join(store_dir, pred_name)
    return pred_path


def get_paths(entry):
    store_id, dataset_id, mre_id, pred_id = ids = get_ids(entry)

    store_path = get_store_path(store_id)
    dataset_path = get_dataset_path(dataset_id)
    mre_path = get_mre_path(mre_id)
    pred_path = get_pred_path(pred_id)

    return store_path, dataset_path, mre_path, pred_path