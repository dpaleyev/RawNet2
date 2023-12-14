import torch
import typing as tp

def collate_fn(batch: tp.List[dict]):
    audios = torch.cat([item['audio'][0] for item in batch], dim=0)
    lables = torch.LongTensor([item['label'] for item in batch])
    return {
        'audio': audios,
        'label': lables
    }
