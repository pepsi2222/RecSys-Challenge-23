from recstudio.utils import *
from recstudio import quickstart
import os

# task = 'is_clicked' 
task = 'is_installed'
fine_tune = False

if __name__ == '__main__':
    parser = get_default_parser()
    args, command_line_args = parser.parse_known_args()
    parser = add_model_arguments(parser, args.model)
    command_line_conf = parser2nested_dict(parser, command_line_args)

    model_class, model_conf = get_model(args.model)
    model_conf = deep_update(model_conf, command_line_conf)
    if os.getcwd().endswith('RecStudio'):
        sh = 'yaml_adjust.sh'
    else:
        sh = 'RecStudio/yaml_adjust.sh'
    os.system(f"bash {sh} {model_class.__name__ in ['HardShare', 'MMoE', 'PLE', 'AITM']} {task} START {fine_tune}")
    quickstart.run(args.model, args.dataset, model_config=model_conf, data_config_path=args.data_config_path)
    os.system(f"bash {sh} {model_class.__name__ in ['HardShare', 'MMoE', 'PLE', 'AITM']} {task} END {fine_tune}")
