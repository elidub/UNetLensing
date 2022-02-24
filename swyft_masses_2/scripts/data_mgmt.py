import os

def print_existance(file, path):
    print(f'{file} {path} exists!') if os.path.exists(path) else print(f'{file} does not exist!')

def get_store_path(store_id, store_dir = '/nfs/scratch/eliasd/store'):
    store_name = f'store_{store_id}.zarr'
    store_path = os.path.join(store_dir, store_name)
    print_existance('Store', store_path)
    return store_path
                     
def get_dataset_path(dataset_id, store_dir = '/nfs/scratch/eliasd/dataset'):
    dataset_name = f'dataset_{dataset_id}.zarr'
    dataset_path = os.path.join(store_dir, dataset_name)
    print_existance('Dataset', dataset_path)
    return dataset_path

def get_mre_path(mre_id, mre_dir = '../data/mre'):
    mre_name = f'mre_{mre_id}.zarr'
    mre_path = os.path.join(store_dir, mre_name)
    print_existance('mre', mre_path)
    return mre_path

def get_pred_path(pred_id, pred_dir = '../data/pred'):
    pred_name = f'pred_{pred_id}.zarr'
    pred_path = os.path.join(store_dir, pred_name)
    print_existance('pred', pred_path)
    return pred_path