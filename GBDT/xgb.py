import xgboost as xgb
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

parser.add_argument('--gpu', type=str, default='6')
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--infer', type=bool, default=True)
parser.add_argument('--infer_all', type=bool, default=False)

parser.add_argument('--weekday', type=bool, default=True)

# 640
parser.add_argument('--md', type=int, default=2)
parser.add_argument('--mcw', type=int, default=2000)
parser.add_argument('--gamma', type=float, default=0.)
parser.add_argument('--csb', type=float, default=1)
parser.add_argument('--ss', type=float, default=0.8)
parser.add_argument('--rl', type=float, default=10)
parser.add_argument('--ra', type=float, default=0.)
parser.add_argument('--spw', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.1)

# 650
# parser.add_argument('--md', type=int, default=3)
# parser.add_argument('--mcw', type=int, default=3000)
# parser.add_argument('--gamma', type=float, default=0.)
# parser.add_argument('--csb', type=float, default=0.9)
# parser.add_argument('--ss', type=float, default=0.7)
# parser.add_argument('--rl', type=float, default=10)
# parser.add_argument('--ra', type=float, default=0.)
# parser.add_argument('--spw', type=float, default=1)
# parser.add_argument('--lr', type=float, default=0.1)

# all
# parser.add_argument('--md', type=int, default=3)
# parser.add_argument('--mcw', type=int, default=2000)
# parser.add_argument('--gamma', type=float, default=0.)
# parser.add_argument('--csb', type=float, default=0.2)
# parser.add_argument('--ss', type=float, default=0.7)
# parser.add_argument('--rl', type=float, default=10)
# parser.add_argument('--ra', type=float, default=0.01)
# parser.add_argument('--spw', type=float, default=1)
# parser.add_argument('--lr', type=float, default=0.1)

# parser.add_argument('--md', type=int, default=9)
# parser.add_argument('--mcw', type=int, default=1000)
# parser.add_argument('--gamma', type=float, default=0.)
# parser.add_argument('--csb', type=float, default=0.7)
# parser.add_argument('--ss', type=float, default=0.8)
# parser.add_argument('--rl', type=float, default=100)
# parser.add_argument('--ra', type=float, default=0.01)
# parser.add_argument('--spw', type=float, default=1)
# parser.add_argument('--lr', type=float, default=0.1)

# parser.add_argument('--md', type=int, default=3)
# parser.add_argument('--mcw', type=int, default=1000)
# parser.add_argument('--gamma', type=float, default=0.5)
# parser.add_argument('--csb', type=float, default=0.7)
# parser.add_argument('--ss', type=float, default=0.8)
# parser.add_argument('--rl', type=float, default=10000)
# parser.add_argument('--ra', type=float, default=0.001)
# parser.add_argument('--spw', type=float, default=1)
# parser.add_argument('--lr', type=float, default=0.1)

# parser.add_argument('--md', type=int, default=5)
# parser.add_argument('--mcw', type=int, default=1000)
# parser.add_argument('--gamma', type=float, default=0.5)
# parser.add_argument('--csb', type=float, default=0.5)
# parser.add_argument('--ss', type=float, default=0.9)
# parser.add_argument('--rl', type=float, default=10)
# parser.add_argument('--ra', type=float, default=0.001)
# parser.add_argument('--spw', type=float, default=1)
# parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args() 

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


params = {
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'random_state': args.seed,
    'max_depth': args.md, 
    'min_child_weight': args.mcw,   
    'gamma': args.gamma, 
    'colsample_bytree': args.csb,
    'subsample': args.ss,
    'reg_lambda': args.rl,
    'reg_alpha': args.ra,
    'scale_pos_weight': args.spw,
    'learning_rate': args.lr,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor'
}
if args.train:
    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = f"./log/XGBoost/{args.probs}-650/{log_time}.log"
    logger = get_logger(log_path)
    logger.info(f'log saved in {log_path}')
    sys.stdout.write = logger.info
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(params, False))
    logger.info('Loading csv')
    
if args.probs == 'all' or args.probs == 'install':
    cache_path = '/root/autodl-tmp/xingmei/RecSys23/data/install_probs_tvt_0_650.cache'
    if not os.path.exists(cache_path):
        df = pd.read_csv('/root/autodl-tmp/xingmei/RecSys23/data/install_probs_tvt_0_650.csv', sep='\t')
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
            f.close()
    else:
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
            f.close()
    ec = False

if args.probs == 'all':
    cache_path_ = f'/root/autodl-tmp/xingmei/RecSys23/data/click_probs_trn_val_tst.cache'
    if not os.path.exists(cache_path_):
        df_ = pd.read_csv(f'/root/autodl-tmp/xingmei/RecSys23/data/click_probs_trn_val_tst.csv', sep='\t')
        with open(cache_path_, 'wb') as f:
            pickle.dump(df, f)
            f.close()
    else:
        with open(cache_path_, 'rb') as f:
            df_ = pickle.load(f)
            f.close()
            
if args.probs == 'none':
    cache_path = '/root/autodl-tmp/xingmei/RecSys23/data/preprocessed_trn_val_tst.cache'
    path = '/root/autodl-tmp/xingmei/RecSys23/data/preprocessed_trn_val_tst.csv'
    ec = True
    ft = ['q']+['c']*37+['q']+['c']*14+['q']*2+['c']*4+['q']*7+['c']*9
    if args.weekday:
        cache_path = f'/root/autodl-tmp/xingmei/RecSys23/data/preprocessed_trn_val_tst_weekday.cache'
        path = '/root/autodl-tmp/xingmei/RecSys23/data/preprocessed_trn_val_tst_with_weekday.csv'
        ft = ['c']+['q']+['c']*37+['q']+['c']*14+['q']*2+['c']*4+['q']*7+['c']*9
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
    ft = ['q']*len(feats)

if args.train:
    trn_df = df[0:3387880]
    val_df = df[3387880:3387880+97972]
    trn_X, trn_y = trn_df[feats], trn_df['is_installed']
    val_X, val_y = val_df[feats], val_df['is_installed']
    trn_d = xgb.DMatrix(trn_X, trn_y, enable_categorical=ec,
                        feature_names=list(trn_X.columns), 
                        feature_types=ft)
    val_d = xgb.DMatrix(val_X, val_y, enable_categorical=ec,
                        feature_names=list(val_X.columns), 
                        feature_types=ft)
    model = xgb.train(
                params, 
                trn_d, 
                evals=[(trn_d,'train'), (val_d,'eval')],
                num_boost_round=10000, 
                early_stopping_rounds=200,
                verbose_eval=100
            )
    logger.info(f'Best score: {model.best_score} at iteration {model.best_iteration} | Log time {log_time}')
    if not os.path.exists("./saved/XGBoost"):
        os.makedirs("./saved/XGBoost")
    model.save_model(f"./saved/XGBoost/{log_time}.json")
    
if args.infer:
    if args.train:
        save_time = log_time
    else:
        model = xgb.Booster()
        save_time = '2023-06-21-22-11-26'
        model.load_model("./saved/XGBoost/"+save_time+".json")
        
    if not os.path.exists("./predictions/XGBoost"):
        os.makedirs("./predictions/XGBoost")
        
    if not args.infer_all:
        tst_X = df.loc[3387880+97972:, feats]
        tst_d = xgb.DMatrix(tst_X, enable_categorical=ec,
                            feature_names=list(tst_X.columns), 
                            feature_types=ft)
        preds = model.predict(tst_d)
        rowid = pd.read_csv('/root/autodl-tmp/yankai/data/data/tst_rowid.csv')['f_0'].to_list()
        pred_df = pd.DataFrame({
                            'RowId': rowid, 
                            'is_clicked': 0, 
                            'is_installed': preds
                        })
        pred_df.to_csv(f'./predictions/XGBoost/{save_time}.csv', sep='\t', index=False)
    else:
        X = df[feats]
        d = xgb.DMatrix(X, enable_categorical=ec,
                        feature_names=list(X.columns), 
                        feature_types=ft)
        preds = model.predict(d)
        pred_df = pd.DataFrame({'is_installed_prob': preds})
        pred_df.to_csv(f'./predictions/XGBoost/{save_time}is_installed.csv', sep='\t', index=False)