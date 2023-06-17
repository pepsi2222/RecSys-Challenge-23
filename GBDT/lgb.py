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
import numpy as np
from sklearn import metrics


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
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--infer', type=bool, default=True)
parser.add_argument('--fold', type=int, default=None)
parser.add_argument('--seed', type=int, default=2023)

parser.add_argument('--md', type=int, default=-1)
parser.add_argument('--nl', type=int, default=200)
parser.add_argument('--mb', type=int, default=100)
parser.add_argument('--mcs', type=int, default=20)
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--l1', type=float, default=0)
parser.add_argument('--ff', type=float, default=1.0)
parser.add_argument('--bf', type=float, default=1.0)
parser.add_argument('--bfq', type=int, default=5)

parser.add_argument('--lr', type=float, default=0.005)

args = parser.parse_args() 

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


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
    'verbose': 1,
}
if args.train:
    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = f"./log/LightGBM/{log_time}.log"
    logger = get_logger(log_path)
    logger.info(f'log saved in {log_path}')
    sys.stdout.write = logger.info
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(params, False))
    logger.info('Loading csv')
    
cache_path = '/root/autodl-tmp/xingmei/RecSysChallenge23/data/preprocessed_trn_val_tst.cache'
if not os.path.exists(cache_path):
    df = pd.read_csv('/root/autodl-tmp/xingmei/RecSysChallenge23/data/preprocessed_trn_val_tst.csv', sep='\t')
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
        f.close()
else:
    with open(cache_path, 'rb') as f:
        df = pickle.load(f)
        f.close()

field = list(df.columns)
field.pop(-1)
field.pop(-1)

if args.train:
    best_score = 100
    auc = 0
    trn_df = df[0:3387880+97972]
    best_fold = -1
    if args.fold is not None:
        sum_score = 0
        kf = KFold(n_splits=args.fold)  # shuffle=True, random_state=args.seed
        for i, (trn_idx, tst_idx) in enumerate(kf.split(trn_df)):
            logger.info(f'Fold {i+1}: trn size {len(trn_idx)} tst size {len(tst_idx)}')
            trn_X, trn_y = trn_df.loc[trn_idx, field], trn_df.loc[trn_idx, 'is_installed']
            tst_X, tst_y = trn_df.loc[tst_idx, field], trn_df.loc[tst_idx, 'is_installed']
            trn_d = lgb.Dataset(trn_X, trn_y, 
                                feature_name=list(trn_X.columns), 
                                categorical_feature=['f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_60', 'f_61', 'f_62', 'f_63', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79'])
            tst_d = lgb.Dataset(tst_X, tst_y, reference=trn_d,
                                feature_name=list(trn_X.columns), 
                                categorical_feature=['f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_60', 'f_61', 'f_62', 'f_63', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79'])
            
            model = lgb.train(
                        params, 
                        trn_d, 
                        valid_sets=[tst_d],
                        num_boost_round=10000,
                        early_stopping_round=200,
                        feature_name=list(trn_X.columns), 
                        categorical_feature=['f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_60', 'f_61', 'f_62', 'f_63', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79'])
            
            sum_score += model.best_score['valid_0']['binary_logloss']
            if model.best_score['valid_0']['binary_logloss'] < best_score:
                model.save_model(f"./saved/LightGBM/{log_time}_fold{i+1}.json")
                best_score = model.best_score['valid_0']['binary_logloss']
                best_fold = i + 1
        logger.info(f'Best score {best_score} | Log time {log_time} | Fold {best_fold}')
        logger.info(f'Avg score {sum_score / args.fold}')
    else:
        trn_X, trn_y = trn_df[field], trn_df['is_installed']
        trn_d = lgb.Dataset(trn_X, trn_y,
                            feature_name=list(trn_X.columns), 
                            categorical_feature=['f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_60', 'f_61', 'f_62', 'f_63', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79'])

        model = lgb.train(
                    params, 
                    trn_d, 
                    num_boost_round=10000,
                    feature_name=list(trn_X.columns), 
                    categorical_feature=['f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 'f_60', 'f_61', 'f_62', 'f_63', 'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79'])

        # if model.best_score['valid_0']['binary_logloss'] < best_score:
        #     model.model_to_string(f"./saved/LightGBM/{log_time}.json")
        #     if model.best_score['valid_0']['binary_logloss'] < best_score:
        #         best_score = model.best_score['valid_0']['binary_logloss']
        # logger.info(f'Best score {best_score} | Log time {log_time}')
    
if args.infer:
    if args.train:
        save_time = log_time
        fold = best_fold
    else:
        save_time = ''
        fold = ''
        
    if args.fold is not None:
        save_pth = "./saved/LightGBM/"+save_time+"_fold{}.json"
    else:
        save_pth = "./saved/LightGBM/"+save_time+".json"
        
    tst_X = df.loc[3387880+97972:, field]
    if args.fold is not None:
        model = lgb.Booster(model_file=save_pth.format(fold))
    preds = model.predict(tst_X)
    rowid = pd.read_csv('/root/autodl-tmp/xingmei/RecSysChallenge23/data/tst_rowid.csv')['f_0'].to_list()
    pred_df = pd.DataFrame({
                        'RowId': rowid, 
                        'is_clicked': 0, 
                        'is_installed': preds
                    })
    pred_df.to_csv(f'./predictions/LightGBM/{save_time}.csv', sep='\t', index=False)