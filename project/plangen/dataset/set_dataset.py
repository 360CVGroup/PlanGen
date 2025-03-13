import torch
from copy import deepcopy
from .data_toy import Dataset_toy
from .data_hico import Hico_dataset
from torch.utils.data import ConcatDataset

def get_one_dataset(args, data, is_test=False, **kwargs):
    if data == 'data_toy':
        return Dataset_toy(args, is_test=is_test)
    elif data == 'sam':
        return Hico_dataset(args, is_test=is_test, is_sam=True)
    elif data == 'coco':
        return Hico_dataset(args, is_test=is_test, is_coco=True)
    elif data == 'coco_rm':
        return Hico_dataset(args, is_test=is_test, is_coco=True, for_rm=True)
    elif data == 'coco_val17':
        return Hico_dataset(args, is_test=is_test, is_coco=True, coco_split='val17')
    elif data == 'oim':
        return Hico_dataset(args, is_test=is_test, is_oim=True)
    elif data == 'oim_test':
        return Hico_dataset(args, is_test=is_test, is_oim=True, oim_split='test')
    elif data == 'mb':
        return Hico_dataset(args, is_test=is_test, is_mb=True, mb_split='test')
    elif data == 'mb_train':
        return Hico_dataset(args, is_test=is_test, is_mb=True, mb_split='train')
    elif data == 'ultra':
        return Hico_dataset(args, is_test=is_test, is_mb=True, mb_split='ultra')
    elif data == 'hico':
        return Hico_dataset(args, is_test=is_test)
    elif data == 'creati':
        return Hico_dataset(args, is_test=is_test, is_creati=True)
    elif data == '1k':
        return Hico_dataset(args, is_test=is_test, is_creati=True, use_1k=True)
    elif data == '1k_obj':
        return Hico_dataset(args, is_test=is_test, is_creati=True, use_1k=True, obj_level=True)
    elif data == 'layout':
        return Hico_dataset(args, is_test=is_test, is_layout=True)
    elif data == 'gen':
        return Hico_dataset(args, is_test=is_test, is_gen=True)
    elif data == 'edit':
        return Hico_dataset(args, is_test=is_test, is_edit=True)
    elif data == 'edit_coco':
        return Hico_dataset(args, is_test=is_test, is_edit=True, edit_split='edit_coco')
    elif data == 'rm_coco':
        return Hico_dataset(args, is_test=is_test, is_edit=True, edit_split='rm_coco')
    elif data == 'hico_full':
        return Hico_dataset(args, is_test=is_test, tiny_hico_data=False)
    elif data == 'hico_7k':
        return Hico_dataset(args, is_test=is_test, is_7k=True)
    elif data == 'plan_llama':
        return Hico_dataset(args, is_test=is_test, is_plan=True, plan_split='llama')
    elif data == 'plan_r1_qwen':
        return Hico_dataset(args, is_test=is_test, is_plan=True, plan_split='r1_qwen')
    elif data == 'plan_r1':
        return Hico_dataset(args, is_test=is_test, is_plan=True, plan_split='r1')
    elif data == 'plan_qwen':
        return Hico_dataset(args, is_test=is_test, is_plan=True, plan_split='qwen')
    elif data == 'plan_r1':
        return Hico_dataset(args, is_test=is_test, is_plan=True, plan_split='r1')
    elif data == 'hico_full_d':
        return Hico_dataset(args, is_test=is_test, tiny_hico_data=False, can_dropout=True)
    elif data == 'hico_d':
        return Hico_dataset(args, is_test=is_test, can_dropout=True)
    elif data == 'hico_test':
        return Hico_dataset(args, is_test=is_test, tiny_hico_data=True, hico_split='test')
    elif data == 'hico_val':
        return Hico_dataset(args, is_test=is_test, tiny_hico_data=True, hico_split='val')
        import pdb;pdb.set_trace()
    elif isinstance(data, list):
        datas = [] 
        for data_type in data:
            dataset = get_one_dataset(args, data_type, is_test=is_test, **kwargs)
            datas.append(dataset)
        dataset = ConcatDataset(datas)
        return dataset
    else:
        assert False

def loader(dataset, shuffle=True, batch_size=1, dataloader_num_workers=16, pin_memory=False, collate_fn=None, prefetch_factor=None):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )


def set_dataset(args, train_data, test_data, train_batch_size, test_batch_size, tokenizer=None, accelerator=None, collate_fn=None, train_dataset=None, test_dataset=None):
    if train_dataset is None:
        train_dataset = get_one_dataset(args, train_data)
    if test_dataset is None:
        test_dataset  = get_one_dataset(args, test_data, is_test=True)

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    train_dataloader = loader(train_dataset, True, train_batch_size, args.dataloader_num_workers, args.pin_memory, collate_fn=collate_fn)
    test_dataloader  = loader(test_dataset, False, test_batch_size, args.dataloader_num_workers, args.pin_memory, collate_fn=collate_fn)

    return train_dataset, test_dataset, train_dataloader, test_dataloader

def get_dataset(args, data_name, batch_size, tokenizer=None, accelerator=None, collate_fn=None, dataset=None, is_test=False, prefetch_factor=None):
    if dataset is None:
        dataset = get_one_dataset(args, data_name, is_test=is_test)

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate

    dataloader = loader(
        dataset, 
        False if is_test else True, 
        batch_size, 
        args.dataloader_num_workers, 
        args.pin_memory, 
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )

    return dataset, dataloader