import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.data import PersistentDataset
from monai.transforms import Compose, LoadImageD, RandSpatialCropD, ScaleIntensityD, BorderPadD, EnsureChannelFirstD, EnsureTypeD
import torch

class fNIRSDataset(Dataset):
    def __init__(self, data_path, label_path):
        # 加载数据时强制转换为float32
        self.data = np.load(data_path, allow_pickle=True).astype(np.float32)
        
        self.data = self.data.squeeze(1)
        self.labels = np.load(label_path, allow_pickle=True).astype(np.float32)
        
        # 确保标签是二维数组 [samples, 1]
        if self.labels.ndim == 1:
            self.labels = self.labels[:, np.newaxis]
        
        assert len(self.data) == len(self.labels), "数据与标签数量不匹配！"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 处理fNIRS数据
        fnirs_data = self.data[idx]
        if isinstance(fnirs_data, np.ndarray):
            fnirs_tensor = torch.from_numpy(fnirs_data.copy()).float()
        else:
            fnirs_tensor = torch.tensor(fnirs_data, dtype=torch.float32)
        
        # 处理标签数据
        label_data = self.labels[idx]
        if isinstance(label_data, np.ndarray):
            label_tensor = torch.from_numpy(label_data.copy()).float()
        else:
            # 处理标量值
            label_tensor = torch.tensor(label_data, dtype=torch.float32).unsqueeze(0)  # 添加维度
        
        return {
            "input_data": fnirs_tensor,
            "label": label_tensor,
            "subject": 0
        }

def get_trans_fnirs():
    return Compose([
        EnsureChannelFirstD(keys="input_data"),
        ScaleIntensityD(keys="input_data"),
        RandSpatialCropD(keys="input_data", roi_size=[52, 375]),
        EnsureTypeD(keys="input_data", dtype=torch.float32)
    ])

def create_dataloaders(data_paths, batch_size=32, num_workers=12):
    train_dataset = fNIRSDataset(data_paths['train_data'], data_paths['train_labels'])
    valid_dataset = fNIRSDataset(data_paths['valid_data'], data_paths['valid_labels'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, valid_loader
