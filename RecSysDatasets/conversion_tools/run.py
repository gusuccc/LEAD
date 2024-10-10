# @Time   : 2020/9/18
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


import argparse
import importlib

from src.utils import dataset2class, click_dataset, multiple_dataset, multiple_item_features


def run(dataset, input_path, output_path, interaction_type=None, duplicate_removal=False, item_feature_name='none', convert_inter=False, convert_item=False, convert_user=False):
    assert input_path is not None, 'input_path can not be None, please specify the input_path'
    assert output_path is not None, 'output_path can not be None, please specify the output_path'

    input_args = [input_path, output_path]
    dataset_class_name = dataset2class[dataset.lower()]
    dataset_class = getattr(importlib.import_module('src.extended_dataset'), dataset_class_name)
    if dataset_class_name in multiple_dataset:
        input_args.append(interaction_type)
    if dataset_class_name in click_dataset:
        input_args.append(duplicate_removal)
    if dataset_class_name in multiple_item_features:
        input_args.append(item_feature_name)
    datasets = dataset_class(*input_args)

    if convert_inter:
        datasets.convert_inter()
    if convert_item:
        datasets.convert_item()
    if convert_user:
        datasets.convert_user()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m')
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--interaction_type', type=str, default=None)
    parser.add_argument('--duplicate_removal', action='store_true')
    parser.add_argument('--item_feature_name', type=str, default='none')
    parser.add_argument('--convert_inter', action='store_true')
    parser.add_argument('--convert_item', action='store_true')
    parser.add_argument('--convert_user', action='store_true')

    args = parser.parse_args()

    run(args.dataset, args.input_path, args.output_path, args.interaction_type, args.duplicate_removal, args.item_feature_name, args.convert_inter, args.convert_item, args.convert_user)


if __name__ == '__main__':
    run(dataset='amazon_books',
         input_path='/data7_8T/ycx/tmp/Recbole_STRec/RecBole-GNN/data/steam_cg',
         output_path='/data7_8T/ycx/tmp/Recbole_STRec/RecBole-GNN/data/steam_cg',
         convert_inter=True,
         convert_item=False,
         convert_user=False)
