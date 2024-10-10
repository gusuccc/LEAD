import argparse
import os
from recbole_gnn.quick_start import run_recbole_gnn

# 定义模型和对应的配置文件
model_config_map = {
    'LightGCN': './config/cg/LightGCN.yaml',
}

# 数据集列表
dataset_list = [
    'Amazon_Books_cg',
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config files')

    args = parser.parse_args()

    for model, config_file in model_config_map.items():
        for dataset in dataset_list:
            print(f'Running model: {model} with dataset: {dataset} using config: {config_file}')
            run_recbole_gnn(model=model, dataset=dataset, config_file_list=[config_file])



