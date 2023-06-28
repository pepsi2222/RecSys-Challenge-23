import lightgbm as lgb
import pandas as pd
import argparse
import time
from sklearn.model_selection import KFold
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
parser.add_argument('--probs', type=str, default='install', choices=['install', 'all', 'none'])
parser.add_argument('--seed', type=int, default=2023)

parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--fold', type=int, default=None)
nbr = 40

parser.add_argument('--infer', type=bool, default=True)
parser.add_argument('--infer_all', type=bool, default=False)

parser.add_argument('--weekday', type=bool, default=True)

parser.add_argument('--md', type=int, default=-1)
parser.add_argument('--nl', type=int, default=1000)
parser.add_argument('--mb', type=int, default=100)
parser.add_argument('--mcs', type=int, default=100)
parser.add_argument('--l2', type=float, default=100)
parser.add_argument('--l1', type=float, default=0.1)
parser.add_argument('--ff', type=float, default=0.6)
parser.add_argument('--bf', type=float, default=1)
parser.add_argument('--bfq', type=int, default=100)

parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args() 


params = {
    'objective': 'regression',
    'boosting': 'gbdt',
    'data_sample_strategy': 'bagging',
    'metric':['binary_logloss', 'auc'],
    'seed': args.seed,
    'max_depth': args.md, 
    'num_leaves': args.nl,
    'max_bin': args.mb,
    'min_child_samples': args.mcs,
    'lambda_l1': args.l1,
    'lambda_l2': args.l2,
    'feature_fraction': args.ff,
    'bagging_fraction': args.bf,
    'bagging_freq': args.bfq,
    'learning_rate': args.lr,
    'device_type':'cpu',
    'verbose': -1,
    'early_stopping_round': 200,
}

data_dir = '/root/autodl-tmp/xingmei/RecSysChallenge23/data'

if args.train:
    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = f"./log/LightGBM/{args.probs}/{log_time}.log"
    logger = get_logger(log_path)
    logger.info(f'log saved in {log_path}')
    sys.stdout.write = logger.info
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(params, False))
    
if args.probs == 'all' or args.probs == 'install':
    cache_path = os.path.join(data_dir, 'install_probs_tvt_0.cache')
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


if args.probs == 'all':
    df = pd.concat([df, df_], axis=1)
    
feats = list(df.columns)
feats.remove('is_installed')



if args.train:
    trn_df = df[3387880:3387880+97972]
    trn_df.reset_index(inplace=True)
    val_X = df[:3387880][feats]
    val_y = df[:3387880]['is_installed']
    val_logloss = sum_score = 0
    if args.fold is not None:
        sum_score = 0
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        for i, (trn_idx, tst_idx) in enumerate(kf.split(trn_df)):
            logger.info(f'Fold {i+1}: trn size {len(trn_idx)} tst size {len(tst_idx)}')
            trn_X, trn_y = trn_df.loc[trn_idx, feats], trn_df.loc[trn_idx, 'is_installed']
            tst_X, tst_y = trn_df.loc[tst_idx, feats], trn_df.loc[tst_idx, 'is_installed']

            trn_d = lgb.Dataset(trn_X, trn_y, feature_name=list(trn_X.columns))
            tst_d = lgb.Dataset(tst_X, tst_y, reference=trn_d, feature_name=list(trn_X.columns))
            model = lgb.train(
                        params, 
                        trn_d, 
                        valid_sets=[tst_d],
                        num_boost_round=10000,
                        feature_name=list(trn_X.columns))
            sum_score += model.best_score['valid_0']['binary_logloss']
            preds = model.predict(val_X)
            preds = [0 if i < 0 else 1 if i > 1 else i for i in preds]
            val_logloss += M.log_loss(val_y, preds)
        logger.info(f'Avg score {sum_score / args.fold} | Log time {log_time} | {val_logloss / args.fold}')
    else:
        trn_X, trn_y = trn_df[feats], trn_df['is_installed']
        trn_d = lgb.Dataset(trn_X, trn_y, feature_name=list(trn_X.columns))
        model = lgb.train(
                    params, 
                    trn_d, 
                    num_boost_round=nbr,
                    valid_sets=[trn_d],
                    feature_name=list(trn_X.columns))
        preds = model.predict(val_X)
        preds = [0 if i < 0 else 1 if i > 1 else i for i in preds]
        val_logloss = M.log_loss(val_y, preds)
        logger.info(f"Best score {model.best_score} | Log time {log_time} | {val_logloss} | Iteration {model.best_iteration}")
        
    if not os.path.exists("./saved/LightGBM"):
        os.makedirs("./saved/LightGBM")
    model.save_model(f"./saved/LightGBM/{log_time}.json")
    
if args.infer:
    if args.train:
        save_time = log_time
    else:
        save_time = '2023-06-22-05-06-45'
        model = lgb.Booster(model_file="./saved/LightGBM/"+save_time+".json")
        
    if not os.path.exists("./predictions/LightGBM"):
            os.makedirs("./predictions/LightGBM")
    
    if not args.infer_all:
        tst_X = df.loc[3387880+97972:, feats]
        preds = model.predict(tst_X)
        rowid = pd.read_csv('/root/autodl-tmp/xingmei/RecSysChallenge23/data/tst_rowid.csv')['f_0'].to_list()
        pred_df = pd.DataFrame({
                            'RowId': rowid, 
                            'is_clicked': 0, 
                            'is_installed': preds
                        })
        pred_df.to_csv(f'./predictions/LightGBM/{save_time}.csv', sep='\t', index=False)
    else:
        X = df[feats]
        preds = model.predict(X)
        pred_df = pd.DataFrame({'is_installed_prob': preds})
        pred_df.to_csv(f'./predictions/LightGBM/{save_time}is_installed.csv', sep='\t', index=False)