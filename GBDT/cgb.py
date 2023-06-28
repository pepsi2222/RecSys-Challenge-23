from catboost import Pool, CatBoostClassifier
import pandas as pd
import argparse
import time
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
parser.add_argument('--probs', type=str, default='none', choices=['install', 'all', 'none'])
parser.add_argument('--seed', type=int, default=2023)

parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--train_with_val', type=bool, default=False)   # should motify iterations
iterations = 250

parser.add_argument('--infer', type=bool, default=True)
parser.add_argument('--infer_all', type=bool, default=True)

parser.add_argument('--weekday', type=bool, default=True)

parser.add_argument('--md', type=int, default=9)
parser.add_argument('--mb', type=int, default=100)
parser.add_argument('--mcs', type=int, default=100)
parser.add_argument('--l2', type=float, default=1000)
# parser.add_argument('--l1', type=float, default=0.1)
# parser.add_argument('--ff', type=float, default=0.6)
# parser.add_argument('--bf', type=float, default=1.0)
# parser.add_argument('--bfq', type=int, default=5)
# parser.add_argument('--ml', type=int, default=12)
# parser.add_argument('--ss', type=float, default=0.8)
# parser.add_argument('--rsm', type=float, default=0.7)

parser.add_argument('--lr', type=float, default=0.1)

args = parser.parse_args() 
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

params = {
    'iterations': iterations if args.train_with_val else 10000,
    'random_state': args.seed,
    'depth': args.md, 
    'min_child_samples': args.mcs,
    'reg_lambda': args.l2,
    'learning_rate': args.lr,
    'task_type': 'GPU',
    'metric_period': 100,
    # 'max_leaves': args.ml,
    # 'subsample': args.ss,
    # 'rsm': args.rsm
}

# params={
#     'loss_function': 'Logloss', # 损失函数，取值RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE, Poisson。默认Logloss。
#     'custom_loss': 'AUC', # 训练过程中计算显示的损失函数，取值Logloss、CrossEntropy、Precision、Recall、F、F1、BalancedAccuracy、AUC等等
#     'eval_metric': 'AUC', # 用于过度拟合检测和最佳模型选择的指标，取值范围同custom_loss
#     'iterations': 10000, # 最大迭代次数，默认500. 别名：num_boost_round, n_estimators, num_trees
#     'learning_rate': args.lr, # 学习速率,默认0.03 别名：eta
#     'random_seed': args.seed, # 训练的随机种子，别名：random_state
#     'l2_leaf_reg': args.l2, # l2正则项，别名：reg_lambda
#     'bootstrap_type': 'Bernoulli', # 确定抽样时的样本权重，取值Bayesian、Bernoulli(伯努利实验)、MVS(仅支持cpu)、Poisson(仅支持gpu)、No（取值为No时，每棵树为简单随机抽样）;默认值GPU下为Bayesian、CPU下为MVS
# #     'bagging_temperature': 0,  # bootstrap_type=Bayesian时使用,取值为1时采样权重服从指数分布；取值为0时所有采样权重均等于1。取值范围[0，inf)，值越大、bagging就越激进
#     'subsample': 0.95, # 样本采样比率（行采样）
#     'sampling_frequency': 'PerTree', # 采样频率，取值PerTree（在构建每棵新树之前采样）、PerTreeLevel（默认值，在子树的每次分裂之前采样）；仅支持CPU
#     'use_best_model': True, # 让模型使用效果最优的子树棵树/迭代次数，使用验证集的最优效果对应的迭代次数（eval_metric：评估指标，eval_set：验证集数据），布尔类型可取值0，1（取1时要求设置验证集数据）
#     'best_model_min_trees': 50, # 最少子树棵树,和use_best_model一起使用
#     'depth': args.md,  # 树深，默认值6
#     'grow_policy': 'SymmetricTree', # 子树生长策略，取值SymmetricTree（默认值，对称树）、Depthwise（整层生长，同xgb）、Lossguide（叶子结点生长，同lgb）
#     'min_data_in_leaf': args.mcs, # 叶子结点最小样本量
# #     'max_leaves': 12, # 最大叶子结点数量
#     'one_hot_max_size': 4, # 对唯一值数量<one_hot_max_size的类别型特征使用one-hot编码
#     # 'rsm': 0.9, # 列采样比率，别名colsample_bylevel 取值（0，1],默认值1
#     'input_borders': None, # 特征数据边界（最大最小边界）、会影响缺失值的处理（nan_mode取值Min、Max时），默认值None、在训练时特征取值的最大最小值即为特征值边界
#     'boosting_type': 'Ordered', # 提升类型，取值Ordered（catboost特有的排序提升，在小数据集上效果可能更好，但是运行速度较慢）、Plain（经典提升）
#     'max_ctr_complexity': 4, # 分类特征交叉的最高阶数，默认值4
#     'logging_level':'Verbose', # 模型训练过程的信息输出等级，取值Silent（不输出信息）、Verbose（默认值，输出评估指标、已训练时间、剩余时间等）、Info（输出额外信息、树的棵树）、Debug（debug信息）
#     'metric_period': 200, # 计算目标值、评估指标的频率，默认值1、即每次迭代都输出目标值、评估指标
#     'border_count': 254, # 数值型特征的分箱数，别名max_bin，取值范围[1,65535]、默认值254（CPU下), # 设置提前停止训练，在得到最佳的评估结果后、再迭代n（参数值为n）次停止训练，默认值不启用
#     'feature_border_type': 'GreedyLogSum', # 数值型特征的分箱方法，取值Median、Uniform、UniformAndQuantiles、MaxLogSum、MinEntropy、GreedyLogSum（默认值）
#     'task_type': 'GPU',
# }

data_dir = '/root/autodl-tmp/xingmei/RecSysChallenge23/data'

if args.train:
    log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_path = f"./log/CatBoost/{args.probs}/{log_time}.log"
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
    cf = ['weekday', 
          'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_8', 'f_9', 'f_10', 
          'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 
          'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 
          'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 
          'f_40', 'f_41', 'f_42', 'f_44', 'f_45', 'f_46', 'f_47', 'f_48', 'f_49', 
          'f_50', 'f_51', 'f_52', 'f_53', 'f_54', 'f_55', 'f_56', 'f_57', 
          'f_60', 'f_61', 'f_62', 'f_63', 
          'f_71', 'f_72', 'f_73', 'f_74', 'f_75', 'f_76', 'f_77', 'f_78', 'f_79']


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
    model = CatBoostClassifier(**params)
    
    if not args.train_with_val:
        trn_df = df[0:3387880+50000]
        val_df = df[3387880+50000:3387880+97972+50000]
        trn_X, trn_y = trn_df[feats], trn_df['is_installed']
        val_X, val_y = val_df[feats], val_df['is_installed']
    
        trn_d = Pool(trn_X, trn_y, cat_features=cf)
        tst_d = Pool(val_X, val_y, cat_features=cf)
            
        model.fit(
                trn_d, 
                use_best_model=True, 
                verbose=True,
                early_stopping_rounds=200,
                eval_set=tst_d
                )
    else:
        trn_df = df[:3387880+97972]
        trn_X, trn_y = trn_df[feats], trn_df['is_installed']
        trn_d = Pool(trn_X, trn_y, cat_features=cf)    
        model.fit(
                trn_d, 
                use_best_model=True, 
                verbose=True,
                early_stopping_rounds=200,
                eval_set=trn_d
                )
    
    logger.info(f'Best score {model.get_best_score()} | Log time {log_time}')
    if not os.path.exists("./saved/CatBoost"):
        os.makedirs("./saved/CatBoost")
    model.save_model(f"./saved/CatBoost/{log_time}.model")
    
if args.infer:
    if args.train:
        save_time = log_time
    else:
        save_time = '2023-06-20-20-28-27'
        save_pth = "./saved/CatBoost/"+save_time+".model"
        model = CatBoostClassifier()
        model.load_model(save_pth)
        
    if not os.path.exists("./predictions/CatBoost"):
        os.makedirs("./predictions/CatBoost")
        
    if not args.infer_all:
        tst_X = df.loc[3387880+97972:, feats]
        tst_d = Pool(tst_X, cat_features=cf)
        preds = model.predict_proba(tst_d)
        rowid = pd.read_csv('/root/autodl-tmp/xingmei/RecSysChallenge23/data/tst_rowid.csv')['f_0'].to_list()
        pred_df = pd.DataFrame({
                            'RowId': rowid, 
                            'is_clicked': 0, 
                            'is_installed': preds[:, 1]
                        })
        pred_df.to_csv(f'./predictions/CatBoost/{save_time}.csv', sep='\t', index=False)
    else:
        X = df[feats]
        d = Pool(X, cat_features=cf)
        preds = model.predict_proba(d)
        pred_df = pd.DataFrame({'is_installed_prob': preds[:, 1]})
        pred_df.to_csv(f'./predictions/CatBoost/{save_time}is_installed.csv', sep='\t', index=False)