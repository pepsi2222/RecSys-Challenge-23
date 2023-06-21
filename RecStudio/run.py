from recstudio.utils import *
from recstudio import quickstart
import os

# task = 'click' 
task = 'install'
with_weekday = True
fine_tune = False
model = 'LR3'
val536066 = True


# ckpt_name = '2023-05-20-22-31-04.ckpt'
# ckpt_name = '2023-06-15-23-25-04.ckpt'

if __name__ == '__main__':
    parser = get_default_parser(task, with_weekday, fine_tune, model, val536066)
    args, command_line_args = parser.parse_known_args()
    parser = add_model_arguments(parser, args.model)
    command_line_conf = parser2nested_dict(parser, command_line_args)

    model_class, model_conf = get_model(args.model)
    model_conf = deep_update(model_conf, command_line_conf)
    # if os.getcwd().endswith('RecStudio'):
    #     sh = 'yaml_adjust.sh'
    # else:
    #     sh = 'RecStudio/yaml_adjust.sh'
    # os.system(f"bash {sh} {model_class.__name__ in ['HardShare', 'MMoE', 'PLE', 'AITM']} {task} START {fine_tune}")
    quickstart.run(args.model, args.dataset, fine_tune, model_config=model_conf, data_config_path=args.data_config_path)
    # quickstart.pred(args.model, args.dataset, ckpt_name, model_config=model_conf, data_config_path=args.data_config_path)
    # os.system(f"bash {sh} {model_class.__name__ in ['HardShare', 'MMoE', 'PLE', 'AITM']} {task} END {fine_tune}")
