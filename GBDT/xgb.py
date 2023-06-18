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

parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--fold', type=int, default=None)
parser.add_argument('--infer', type=bool, default=True)

parser.add_argument('--md', type=int, default=3)
parser.add_argument('--mcw', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.)
parser.add_argument('--csb', type=float, default=0.8)
parser.add_argument('--ss', type=float, default=0.95)
parser.add_argument('--rl', type=float, default=0)
parser.add_argument('--ra', type=float, default=10)
parser.add_argument('--spw', type=float, default=1)
parser.add_argument('--lr', type=float, default=0.1)

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
    log_path = f"./log/XGBoost/{log_time}.log"
    logger = get_logger(log_path)
    logger.info(f'log saved in {log_path}')
    sys.stdout.write = logger.info
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(params, False))
    logger.info('Loading csv')
    
if args.probs == 'all' or args.probs == 'install':
    cache_path = f'/root/autodl-tmp/xingmei/RecSys23/data/install_probs_trn_val_tst.cache'
    if not os.path.exists(cache_path):
        df = pd.read_csv(f'/root/autodl-tmp/xingmei/RecSys23/data/install_probs_trn_val_tst.csv', sep='\t')
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
            f.close()
    else:
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
            f.close()

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
    cache_path_ = f'/root/autodl-tmp/xingmei/RecSys23/data/preprocessed_trn_val_tst.cache'
    if not os.path.exists(cache_path_):
        df_ = pd.read_csv(f'/root/autodl-tmp/xingmei/RecSys23/data/preprocessed_trn_val_tst.csv', sep='\t')
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
if args.probs == 'all' or args.probs == 'install':
    for i in [
                'p_install_LorentzFM', 
                'p_install_DeepFM', 
                'p_install_EDCN', 
                'p_install_FM', 
                'p_install_LR', 
                'p_install_PNN', 
                'p_install_xDeepFM',
                'p_install_DeepCrossing'
            ]:
        feats.remove(i)

if args.train:
    trn_df = df[0:3387880+97972]
    # trn_df = df[3387880:3387880+97972]
    # trn_df.reset_index(inplace=True)
    if args.fold is not None:
        sum_score = 0
        kf = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)
        for i, (trn_idx, tst_idx) in enumerate(kf.split(trn_df)):
            logger.info(f'Fold {i+1}: trn size {len(trn_idx)} tst size {len(tst_idx)}')
            trn_X, trn_y = trn_df.loc[trn_idx, feats], trn_df.loc[trn_idx, 'is_installed']
            tst_X, tst_y = trn_df.loc[tst_idx, feats], trn_df.loc[tst_idx, 'is_installed']
            trn_d = xgb.DMatrix(trn_X, trn_y, enable_categorical=True, 
                                feature_names=list(trn_X.columns), 
                                feature_types=['q']*len(feats))
            tst_d = xgb.DMatrix(tst_X, tst_y, enable_categorical=True,
                                feature_names=list(trn_X.columns), 
                                feature_types=['q']*len(feats))
            model = xgb.train(
                        params, 
                        trn_d, 
                        evals=[(trn_d,'train'), (tst_d,'eval')],
                        num_boost_round=10000, 
                        early_stopping_rounds=200,
                        verbose_eval=100
                    )
            
            logger.info(f'Best score: {model.best_score} at iteration {model.best_iteration}')
            sum_score += model.best_score
        logger.info(f'Avg score {sum_score / args.fold} | Log time {log_time}')
    else:
        trn_X, trn_y = trn_df[feats], trn_df['is_installed']
        trn_d = xgb.DMatrix(trn_X, trn_y, enable_categorical=True,
                            feature_names=list(trn_X.columns), 
                            feature_types=['q']*len(feats))
        model = xgb.train(
                    params, 
                    trn_d, 
                    evals=[(trn_d,'train')],
                    num_boost_round=10000, 
                    early_stopping_rounds=200,
                    verbose_eval=100
                )
        logger.info(f'Best score: {model.best_score} at iteration {model.best_iteration}')
        if not os.path.exists("./saved/XGBoost"):
            os.makedirs("./saved/XGBoost")
        model.save_model(f"./saved/XGBoost/{log_time}.json")
        logger.info(f'Best score {model.best_score} | Log time {log_time}')
    
if args.infer:
    if args.train:
        save_time = log_time
    else:
        model = xgb.Booster(params)
        save_time = '2023-06-18-16-34-37'
        model.load_model("./saved/XGBoost/"+save_time+".json")
    tst_df = df[3387880+97972:]
    tst_X = tst_df[feats]
    tst_d = xgb.DMatrix(tst_X, enable_categorical=True,
                        feature_names=list(tst_X.columns), 
                        feature_types=['q']*len(feats))
    preds = model.predict(tst_d)
    rowid = pd.read_csv('/root/autodl-tmp/yankai/data/data/tst_rowid.csv')['f_0'].to_list()
    pred_df = pd.DataFrame({
                        'RowId': rowid, 
                        'is_clicked': 0, 
                        'is_installed': preds
                    })
    if not os.path.exists("./predictions/XGBoost"):
            os.makedirs("./predictions/XGBoost")
    pred_df.to_csv(f'./predictions/XGBoost/{save_time}.csv', sep='\t', index=False)