"""
データセットとトランスフォームを管理するユーティリティ
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional


class DataManager:
    """データセットとデータローダーを管理するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config.get('dataset', {}).get('data_dir', '../data')
        self.batch_size = config.get('dataset', {}).get('batch_size', 64)
        self.num_workers = config.get('dataset', {}).get('num_workers', 2)
        
        # デフォルトの正規化パラメータ（CIFAR-10）
        self.default_mean = (0.4914, 0.4822, 0.4465)
        self.default_std = (0.2470, 0.2435, 0.2616)
        
        self._setup_transforms()
        self._setup_datasets()
        self._setup_dataloaders()
    
    def _setup_transforms(self) -> None:
        """トランスフォームを設定"""
        train_config = self.config.get('dataset', {}).get('train_transform', {})
        test_config = self.config.get('dataset', {}).get('test_transform', {})
        
        self.train_transform = self._create_train_transform(train_config)
        self.test_transform = self._create_test_transform(test_config)
    
    def _create_train_transform(self, config: Dict[str, Any]) -> transforms.Compose:
        """訓練用トランスフォームを作成"""
        transform_list = []
        
        # RandomCrop
        crop_size = config.get('random_crop', 32)
        padding = config.get('padding', 4)
        transform_list.append(transforms.RandomCrop(crop_size, padding=padding))
        
        # RandomHorizontalFlip
        if config.get('random_horizontal_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # ToTensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        normalize_config = config.get('normalize', {})
        mean = normalize_config.get('mean', self.default_mean)
        std = normalize_config.get('std', self.default_std)
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def _create_test_transform(self, config: Dict[str, Any]) -> transforms.Compose:
        """テスト用トランスフォームを作成"""
        transform_list = [transforms.ToTensor()]
        
        # Normalize
        normalize_config = config.get('normalize', {})
        mean = normalize_config.get('mean', self.default_mean)
        std = normalize_config.get('std', self.default_std)
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def _setup_datasets(self) -> None:
        """データセットを設定"""
        dataset_name = self.config.get('dataset', {}).get('name', 'CIFAR10')
        
        if dataset_name.upper() == 'CIFAR10':
            self.trainset = torchvision.datasets.CIFAR10(
                self.data_dir, train=True, download=True,
                transform=self.train_transform
            )
            self.testset = torchvision.datasets.CIFAR10(
                self.data_dir, train=False, download=True,
                transform=self.test_transform
            )
        else:
            raise ValueError(f"サポートされていないデータセット: {dataset_name}")
        
        self.num_classes = len(self.trainset.classes)
    
    def _setup_dataloaders(self) -> None:
        """データローダーを設定"""
        self.trainloader = DataLoader(
            self.trainset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True, 
            drop_last=True
        )
        
        self.testloader = DataLoader(
            self.testset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            drop_last=False
        )
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """データローダーを取得"""
        return self.trainloader, self.testloader
    
    def get_num_classes(self) -> int:
        """クラス数を取得"""
        return self.num_classes


@torch.no_grad()
def update_bn_on_device(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> None:
    """バッチ正規化の統計を更新"""
    model.train()
    for inputs, _ in loader:
        inputs = inputs.to(device)
        model(inputs)
