import torch
import typing as tp

def collate_fn(batch: tp.List[dict]):
    audios = torch.stack([item['audio'][0] for item in batch])
    lables = torch.LongTensor([item['label'] for item in batch])
    return {
        'audio': audios,
        'label': lables
    }
