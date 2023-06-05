import os, time, torch
from typing import *
from recstudio.utils import *
import logging
from recstudio import LOG_DIR

def run(model: str, dataset: str, fine_tune: bool, model_config: Dict=None, data_config: Dict=None, model_config_path: str=None, data_config_path: str=None, verbose=True, **kwargs):
    model_class, model_conf = get_model(model)

    if model_config_path is not None:
        if isinstance(model_config_path, str):
            model_conf = deep_update(model_conf, parser_yaml(model_config_path))
        else:
            raise TypeError(f"expecting `model_config_path` to be str, while get {type(model_config_path)} instead.")

    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf = deep_update(model_conf, model_config)
        else:
            raise TypeError(f"expecting `model_config` to be Dict, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf = deep_update(model_conf, kwargs)

    if fine_tune:
        log_path = time.strftime(f"{model}/finetune/%Y-%m-%d-%H-%M-%S.log", time.localtime())
    else:
        log_path = time.strftime(f"{model}/{dataset}/%Y-%m-%d-%H-%M-%S.log", time.localtime())
    logger = get_logger(log_path)
    torch.set_num_threads(model_conf['train']['num_threads'])

    if not verbose:
        import logging
        logger.setLevel(logging.ERROR)

    logger.info("Log saved in {}.".format(os.path.abspath(os.path.join(LOG_DIR, log_path))))
    model = model_class(model_conf)
    dataset_class = model_class._get_dataset_class()

    data_conf = {}
    if data_config_path is not None:
        if isinstance(data_config_path, str):
            # load dataset config from file
            conf = parser_yaml(data_config)
            data_conf.update(conf)
        else:
            raise TypeError(f"expecting `data_config_path` to be str, while get {type(data_config_path)} instead.")

    if data_config is not None:
        if isinstance(data_config, dict):
            # update config with given dict
            data_conf.update(data_config)
        else:
            raise TypeError(f"expecting `data_config` to be Dict, while get {type(data_config)} instead.")

    data_conf.update(model_conf['data'])    # update model-specified config

    datasets = dataset_class(name=dataset, config=data_conf).build(**model_conf['data'])
    logger.info(f"{datasets[0]}")
    logger.info(f"\n{set_color('Model Config', 'green')}: \n\n" + color_dict_normal(model_conf, False))
    model.fit(*datasets[:2], run_mode='light')
    if len(datasets[-1].data_index) > 0:
        save_dir = os.path.join('./predictions', f'{model.__class__.__name__}')
        save_path = os.path.join(save_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + f'{str(model.frating)}.csv')
        
        train_pred_df = model.predict(datasets[0], dataset='train')
        train_pred_df.to_csv(save_path, sep='\t', index=False)
        
        val_pred_df = model.predict(datasets[1], dataset='val')
        val_pred_df.to_csv(save_path, sep='\t', index=False, mode='a', header=0)
        
        tst_pred_df = model.predict(datasets[-1], dataset='val')
        tst_pred_df.to_csv(save_path, sep='\t', index=False, mode='a', header=0)
        
        logger.info(f'Predictions saved in {save_path}')
        

def pred(model: str, dataset: str, ckpt_name: str, model_config: Dict=None, data_config: Dict=None, model_config_path: str=None, data_config_path: str=None, verbose=True, **kwargs):
    model_class, model_conf = get_model(model)

    if model_config_path is not None:
        if isinstance(model_config_path, str):
            model_conf = deep_update(model_conf, parser_yaml(model_config_path))
        else:
            raise TypeError(f"expecting `model_config_path` to be str, while get {type(model_config_path)} instead.")

    if model_config is not None:
        if isinstance(model_config, Dict):
            model_conf = deep_update(model_conf, model_config)
        else:
            raise TypeError(f"expecting `model_config` to be Dict, while get {type(model_config)} instead.")

    if kwargs is not None:
        model_conf = deep_update(model_conf, kwargs)

    log_path = time.strftime(f"{model}/{dataset}/%Y-%m-%d-%H-%M-%S.log", time.localtime())
    logger = get_logger(log_path)
    torch.set_num_threads(model_conf['train']['num_threads'])

    if not verbose:
        import logging
        logger.setLevel(logging.ERROR)

    logger.info("Log saved in {}.".format(os.path.abspath(log_path)))
    model = model_class(model_conf)
    dataset_class = model_class._get_dataset_class()

    data_conf = {}
    if data_config_path is not None:
        if isinstance(data_config_path, str):
            # load dataset config from file
            conf = parser_yaml(data_config)
            data_conf.update(conf)
        else:
            raise TypeError(f"expecting `data_config_path` to be str, while get {type(data_config_path)} instead.")

    if data_config is not None:
        if isinstance(data_config, dict):
            # update config with given dict
            data_conf.update(data_config)
        else:
            raise TypeError(f"expecting `data_config` to be Dict, while get {type(data_config)} instead.")

    data_conf.update(model_conf['data'])    # update model-specified config

    datasets = dataset_class(name=dataset, config=data_conf).build(**model_conf['data'])

    model._init_model(datasets[0])

    # datasets[-1].drop_feat(keep_fields=model.fields)
    # pred_loader = datasets[-1].eval_loader(batch_size=128)
    # model.load_checkpoint("/root/autodl-tmp/yankai/Sharechat-RecSys-Challenge-23/saved/DCNv2/recsys/2023-05-04-23-29-33.ckpt")      
    # model.eval()
    # outputs = model.predict_epoch(pred_loader)
    # pred_df = model.predict_epoch_end(outputs)

    # save_dir = os.path.join('./predictions', f'{model.__class__.__name__}')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + f'{str(model.frating)}.csv')
    # pred_df.to_csv(save_path, sep='\t', index=False)
    
    model.ckpt_path = f'{model.__class__.__name__}/{dataset}/{ckpt_name}'
    save_dir = os.path.join('./predictions', f'{model.__class__.__name__}')
    save_path = os.path.join(save_dir, ckpt_name.replace('.ckpt', '') + f'{str(model.frating)}.csv')
    
    train_pred_df = model.predict(datasets[0], dataset='train')
    train_pred_df.to_csv(save_path, sep='\t', index=False)
    
    val_pred_df = model.predict(datasets[1], dataset='val')
    val_pred_df.to_csv(save_path, sep='\t', index=False, mode='a', header=0)
    
    test_pred_df = model.predict(datasets[-1], dataset='test')
    test_pred_df.to_csv(save_path, sep='\t', index=False, mode='a', header=0)
    test_pred_df.to_csv(save_path.replace('.csv', '_tst.csv'), sep='\t', index=False)
    
    logger.info(f'Predictions saved in {save_path}')

    