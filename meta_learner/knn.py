from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import argparse
import time
import os
import sys
import logging
import re
from collections import OrderedDict
import pickle
import sklearn.metrics as M


class RemoveColorFilter(logging.Filter):
    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True

def add_file_handler(logger: logging.Logger, file_path: str, formatter: logging.Formatter = None):
    log_path = file_path
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    if formatter is None:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RemoveColorFilter())
    logger.addHandler(file_handler)
    return logger

def close_logger(logger: logging.Logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
        
def get_logger(file_path: str = None) -> logging.Logger:
    FORMAT = '[%(asctime)s] %(levelname)s %(message)s'
    logger = logging.getLogger('recstudio')
    # close all handlers if exists for next use
    close_logger(logger)

    formatter = logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if file_path is not None:
        logger = add_file_handler(logger, file_path, formatter)
    return logger

def set_color(log, color, highlight=True, keep=False):
    r"""Set color for log string.

    Args:
        log(str): the
    """
    if keep:
        return log
    color_set = ['black', 'red', 'green',
                 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def color_dict_normal(dict_, keep=True,):
    dict_ = OrderedDict(sorted(dict_.items()))
    color_set = ['green', 'blue', 'cyan', 'pink', 'white']

    def color_kv(k, v, k_f, v_f, depth=1):
        key_color = color_set[depth-1]
        val_color = 'yellow'
        if isinstance(v, dict):
            v_info = ('\n'+'\t'*depth).join([color_kv(k_, v_, '%s', '%s', depth+1) for k_, v_ in v.items()])
            info = (set_color(k_f, key_color, keep=keep) + ":\n"+ "\t"*depth + v_info) % (k)
        else:
            v = str(v)
            info = (set_color(k_f, key_color, keep=keep) + '=' +
                    set_color(v_f, val_color, keep=keep)) % (k, v)
        return info
    info = ('\n').join([color_kv(k, v, '%s', '%s') for k, v in dict_.items()])
    return info


parser = argparse.ArgumentParser()
parser.add_argument('--probs', type=str, default='none', choices=['install', 'all', 'none'])
parser.add_argument('--seed', type=int, default=2023)

parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--train_with_val', type=bool, default=False)   # should motify iterations
iterations = 250

parser.add_argument('--infer', type=bool, default=False)
parser.add_argument('--infer_all', type=bool, default=False)

parser.add_argument('--weekday', type=bool, default=True)

args = parser.parse_args() 

params = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
    'leaf_size': 30,
    'n_jobs': -1
}

data_dir = '/root/autodl-tmp/xingmei/RecSysChallenge23/data'

if args.train:
    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not os.path.exists('./log/KNN/'):
        os.makedirs('./log/KNN/')
    log_path = f"./log/KNN/{args.probs}/{log_time}.log"
    logger = get_logger(log_path)
    logger.info(f'log saved in {log_path}')
    sys.stdout.write = logger.info
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(params, False))
    
if args.probs == 'all' or args.probs == 'install':
    cache_path = os.path.join(data_dir, 'install_probs_trn_val_tst.cache')
    if not os.path.exists(cache_path):
        df = pd.read_csv(os.path.join(data_dir, 'install_probs_trn_val_tst.csv'), sep='\t')
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
            f.close()
    else:
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
            f.close()

if args.probs == 'all':
    cache_path_ = os.path.join(data_dir, 'click_probs_trn_val_tst.cache')
    if not os.path.exists(cache_path_):
        df_ = pd.read_csv(os.path.join(data_dir, 'click_probs_trn_val_tst.csv'), sep='\t')
        with open(cache_path_, 'wb') as f:
            pickle.dump(df, f)
            f.close()
    else:
        with open(cache_path_, 'rb') as f:
            df_ = pickle.load(f)
            f.close()
            
if args.probs == 'none':
    cache_path = os.path.join(data_dir, 'preprocessed_trn_val_tst.cache')
    path = os.path.join(data_dir, 'preprocessed_trn_val_tst.csv')
    if args.weekday:
        cache_path = os.path.join(data_dir, 'preprocessed_trn_val_tst_weekday.cache')
        path = os.path.join(data_dir, 'preprocessed_trn_val_tst_with_weekday.csv')
    if not os.path.exists(cache_path):
        df = pd.read_csv(path, sep='\t')
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
            f.close()
    else:
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
            f.close()


if args.probs == 'all':
    df = pd.concat([df, df_], axis=1)
    
feats = list(df.columns)
feats.remove('is_installed')
if args.probs == 'all' or args.probs == 'install':
    for i in feats:
        if i not in [
                        'p_install_PLE', 
                        'p_install_MMoE', 
                        'p_install_IFM',
                        'p_install_FwFM',
                        'p_install_DCNv2',
                    ]:
            feats.remove(i)

if args.train:
    model = KNeighborsClassifier(**params)
    
    if not args.train_with_val:
        trn_df = df[0:3387880]
        val_df = df[3387880:3387880+97972]
        trn_X, trn_y = trn_df[feats], trn_df['is_installed']
        val_X, val_y = val_df[feats], val_df['is_installed']
            
        model.fit(trn_X, trn_y)
        logloss = M.log_loss(val_y, model.predict_proba(val_X))
        logger.info(f'Best score {logloss} | Log time {log_time}')
    else:
        trn_df = df[:3387880+97972]
        trn_X, trn_y = trn_df[feats], trn_df['is_installed']
        model.fit(trn_X, trn_y)
        
    # if not os.path.exists("./saved/KNN"):
    #     os.makedirs("./saved/KNN")
    # model.save_model(f"./saved/KNN/{log_time}.model")
    
if args.infer:
    if args.train:
        save_time = log_time
    else:
        save_time = '2023-06-20-19-13-18'
        save_pth = "./saved/CatBoost/"+save_time+".model"
        model = KNeighborsClassifier(**params)
        # model.load_model(save_pth)
        
    if not os.path.exists("./predictions/KNN"):
        os.makedirs("./predictions/KNN")
        
    if not args.infer_all:
        tst_X = df.loc[3387880+97972:, feats]
        preds = model.predict_proba(tst_X)
        rowid = pd.read_csv('/root/autodl-tmp/xingmei/RecSysChallenge23/data/tst_rowid.csv')['f_0'].to_list()
        pred_df = pd.DataFrame({
                            'RowId': rowid, 
                            'is_clicked': 0, 
                            'is_installed': preds[:, 1]
                        })
        pred_df.to_csv(f'./predictions/KNN/{save_time}.csv', sep='\t', index=False)
    else:
        X = df[feats]
        preds = model.predict_proba(X)
        pred_df = pd.DataFrame({'is_installed_prob': preds[:, 1]})
        pred_df.to_csv(f'./predictions/KNN/{save_time}is_installed.csv', sep='\t', index=False)