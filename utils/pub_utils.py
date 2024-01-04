import json
import os
from typing import Union

import numpy as np
import pandas as pd
import torch
from pandas.tests.io.excel.test_openpyxl import openpyxl


def dp_nested_concat(tensors, new_tensors, padding_index=-100):
    """
    三维使用动态填充时维度不一样拼接会出错，使用新的方法去覆盖以前的方法
    Args:
        tensors:
        new_tensors:
        padding_index:

    Returns:

    """
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples of tensors.
    """
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(dp_nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def atleast_1d(tensor_or_array: Union[torch.Tensor, np.ndarray]):
    if isinstance(tensor_or_array, torch.Tensor):
        if hasattr(torch, "atleast_1d"):
            tensor_or_array = torch.atleast_1d(tensor_or_array)
        elif tensor_or_array.ndim < 1:
            tensor_or_array = tensor_or_array[None]
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=0):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    # Let's figure out the new shape
    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1]),
                 max(tensor1.shape[2], tensor2.shape[2]))
    if tensor1.shape[2] > tensor2.shape[2]:
        # Now let's fill the result tensor
        result = tensor1.new_full(new_shape, padding_index)
        result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
        result[tensor1.shape[0]:, : tensor2.shape[1], :tensor2.shape[1]] = tensor2
    else:
        # Now let's fill the result tensor
        result = tensor1.new_full(new_shape, padding_index)
        result[: tensor1.shape[0], : tensor1.shape[1], : tensor1.shape[1]] = tensor1
        result[tensor1.shape[0]:, : tensor2.shape[1]] = tensor2
    return result


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    # Let's figure out the new shape
    new_shape = (
        array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1]), max(array1.shape[2], array2.shape[2]))

    # Now let's fill the result tensor
    if array1.shape[2] > array2.shape[2]:
        result = np.full_like(array1, padding_index, shape=new_shape)
        result[: array1.shape[0], : array1.shape[1]] = array1
        result[array1.shape[0]:, : array2.shape[1], :array2.shape[2]] = array2
    else:
        result = np.full_like(array1, padding_index, shape=new_shape)
        result[: array1.shape[0], : array1.shape[1], :array1.shape[2]] = array1
        result[array1.shape[0]:, : array2.shape[1]] = array2
    return result


def save_training_args(path=None, trainer=None):
    """
    保存trainer中的训练超参数
    Args:
        path:读取trainer超参数的路径,
        如果trainer不为None,会读取最好的检查点中的training_args.bin文件，
        如果trianer为空需指定一个training_args.bin文件
        trainer:训练过程中的trainer

    Returns:
    """
    training_args = None
    if path != None:
        args_path = os.path.join(path, 'training_args.bin')
        training_args = torch.load(args_path)
    order = ['model', 'best_model_checkpoint', 'test_score', 'best_metric']
    args_dict = {}
    args_dict['best_model_checkpoint'] = [None]
    args_dict['best_metric'] = [None]
    args_dict['test_score'] = [None]
    args_dict['use_fgm'] = [None]
    # trainer不为空记录训练过程中最好的评估数据
    if trainer != None:
        if trainer.state.best_model_checkpoint != None and trainer.state.best_metric != None:
            best_model_checkpoint = str(os.path.split(trainer.state.best_model_checkpoint)[1])
            best_metric = trainer.state.best_metric
            args_dict['best_model_checkpoint'] = [best_model_checkpoint]
            args_dict['best_metric'] = [best_metric]
        training_args = trainer.args
        args_dict['use_fgm'] = [str(trainer.use_fgm) + ' epsilon:' + str(trainer.fgm.epsilon)]
    # 移除training_args中不需要的参数
    remove_list = ['bf16', 'bf16_full_eval', 'ddp_bucket_cap_mb', 'ddp_find_unused_parameters', 'debug', 'do_eval',
                   'do_predict', 'do_train', 'deepspeed', 'fsdp', 'fsdp_min_num_params', 'hub_model_id',
                   'hub_private_repo', 'hub_strategy', 'hub_token', 'jit_mode_eval', 'length_column_name',
                   'load_best_model_at_end', 'local_process_index', 'local_rank', 'log_level', 'log_level_replica',
                   'log_on_each_node', 'logging_dir', 'logging_first_step', 'logging_nan_inf_filter', 'logging_steps',
                   'logging_strategy', 'mp_parameters', 'n_gpu', 'no_cuda', 'output_dir', 'overwrite_output_dir',
                   'past_index', 'per_gpu_eval_batch_size', 'per_gpu_train_batch_size', 'place_model_on_device',
                   'prediction_loss_only', 'process_index', 'push_to_hub', 'push_to_hub_model_id',
                   'push_to_hub_organization', 'push_to_hub_token', 'ray_scope', 'remove_unused_columns', 'report_to',
                   'resume_from_checkpoint', 'sharded_ddp', 'save_on_each_node', 'should_log', 'should_save',
                   'skip_memory_metrics', 'tf32', 'tpu_metrics_debug', 'tpu_num_cores', 'use_legacy_prediction_loop',
                   'world_size', 'xpu_backend', 'dataloader_num_workers', 'dataloader_pin_memory', 'device',
                   'disable_tqdm', 'greater_is_better', 'gradient_checkpointing', 'group_by_length', 'ignore_data_skip',
                   'include_inputs_for_metrics', 'save_total_limit', 'torchdynamo', 'auto_find_batch_size',
                   'dataloader_num_workers', 'dataloader_pin_memory', 'disable_tqdm', 'use_ipex', 'parallel_mode',
                   'half_precision_backend', 'full_determinism', 'fp16_backend', 'adafactor', 'label_names',
                   'fsdp_transformer_layer_cls_to_wrap', 'ddp_timeout_delta', 'ddp_timeout', 'framework',
                   'use_mps_device']
    # 把training_args转成字典
    for name in dir(training_args):
        value = getattr(training_args, name)
        if not name.startswith('__') and not callable(value) and not name.startswith('_') and name not in remove_list:
            order.append(name)
            args_dict[name] = [value]
    _, name = os.path.split(args_dict.get('run_name')[0])
    args_dict['model'] = [name]
    args_dict.pop('run_name')
    order.remove('run_name')
    # 将字典转为DataFrame
    pf = pd.DataFrame.from_dict(args_dict)
    pf = pf[order]
    print(args_dict)
    if os.path.exists('training_args.xlsx'):
        # 将DataFrame追加excel表中
        df1 = pd.DataFrame(pd.read_excel('training_args.xlsx', sheet_name='Sheet1', engine='openpyxl'))  # 读取原数据文件和表
        df2 = pd.concat([df1, pf])
        df2.to_excel('training_args.xlsx', sheet_name='Sheet1', encoding='utf-8', index=False)
    else:
        # 将DataFrame写入excel表中
        pf.to_excel('training_args.xlsx', encoding='utf-8', index=False)


with open('corpus/type_match.json', 'r', encoding='GBK') as f:
    type_match_dict = json.load(f)

with open('corpus/type_match_v3.json', 'r', encoding='GBK') as f:
    type_match1_dict = json.load(f)

with open('corpus/crime_type.json', 'r', encoding='GBK') as f:
    crime_type_dict = json.load(f)


def regularization_match(predictions_logits, triggers):
    """
    规则匹配，匹配触发词，直接输出触发词类型
    Args:
        triggers: 这条数据中触发词的信息 （触发词，开始位置，结束位置）
        predictions_logits:网络预测的结果，还未处理过的  batch_size*class_num*seq_len*seq_len
    Returns:

    """
    # 把logits转化为标签表
    if not isinstance(predictions_logits, torch.Tensor):
        predictions_logits = torch.tensor(predictions_logits)
    predictions_logits = predictions_logits.cpu().numpy()
    logits_max = np.max(predictions_logits, 1)
    select_labels = np.where(logits_max < 0, 0, 1)
    predictions_labels = (np.argmax(predictions_logits, axis=1) + 1) * select_labels
    # 遍历需要预测的触发词
    for i, one_triggers in enumerate(triggers):
        # 遍历一条数据需要预测的触发词
        for trigger in one_triggers:
            text = trigger[0]
            start = int(trigger[1])
            end = int(trigger[2])
            # 把预测为None的触发词进行过滤
            # 获取该触发词的logits
            logits = predictions_logits[i, :, start, end]
            # 触发词类别0-108
            label = predictions_labels[i, start, end]
            type_ids = []
            # 获取包含该触发词的类别列表
            for key, value in type_match_dict.items():
                if text in value:
                    type_id = int(key.split("_")[1]) - 1
                    type_ids.append(type_id)
            # 如果有类别包含了这个触发词，则把对应位置的标签修改为包含该触发词的所有类别中概率最大的类别，否则不修改
            if len(type_ids) != 0 and label != 0:
                # 该触发词只存在None类别中,直接修改标签为0
                if len(type_ids) == 1 and type_ids[0] == -1:
                    predictions_labels[i, start, end] = 0
                else:  # 该触发词在好几个类别中均有出现
                    # max_logit = 0  # 概率大于0才考虑更改值
                    # max_index = label - 1  # 若所有的概率都不大于0，则最后类别为原来类别
                    # for type_id in type_ids:
                    #     if type_id != -1:
                    #         if max_logit < logits[type_id]:
                    #             max_logit = logits[type_id]
                    #             max_index = type_id
                    # predictions_labels[i, start, end] = max_index + 1
                    predictions_labels[i, start, end] = label
            elif len(type_ids) > 0 and label == 0:
                # if len(type_ids) == 1 and type_ids[0] == -1:
                #     break
                # max_logit = 0  # 概率大于0才考虑更改值
                # max_index = -1  # 若所有的概率都不大于0，则最后类别修改为0
                # for type_id in type_ids:
                #     if type_id != -1:
                #         if max_logit < logits[type_id]:
                #             max_logit = logits[type_id]
                #             max_index = type_id
                # type_id = max_index + 1
                # crime_text = trigger[3]
                # # 该罪名中是否存在该触发词,如果该罪名中存在该触发词则修改类型
                # if [text, type_id] in crime_type_dict[crime_text]:
                #     predictions_labels[i, start, end] = type_id
                # 如果标签为0，则去type_match1字典中匹配，与type_match区别是None类为空，且去过重
                for key, value in type_match1_dict.items():
                    if text in value:  # 匹配上了再去判断该罪名中是否存在该触发词
                        #type_id = max_index + 1
                        type_id = int(key.split("_")[1])
                        # predictions_labels[i, start, end] = type_id
                        crime_text = trigger[3]
                        # 如果该罪名中存在该触发词则修改类型
                        if [text, type_id] in crime_type_dict[crime_text]:
                            predictions_labels[i, start, end] = type_id
    return predictions_labels


# 全局变量模块

class GlobalVariable:
    event_text_input = None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
