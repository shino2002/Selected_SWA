"""
モデル、オプティマイザー、スケジューラーの作成を管理するファクトリー
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from typing import Dict, Any, Tuple, Optional
from torch.optim.swa_utils import AveragedModel, SWALR

from swa_utils import StepwiseConditionalSWA


class ModelFactory:
    """モデル、オプティマイザー、スケジューラーを作成するファクトリークラス"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.num_classes = config.get('model', {}).get('num_classes', 10)

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        """値をfloatに変換（失敗時はデフォルト）"""
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        """値をintに変換（失敗時はデフォルト）"""
        try:
            if value is None:
                return int(default)
            # 例えば"0.4"のような文字列はfloat→intで安全化
            if isinstance(value, str) and any(ch in value for ch in ['.', 'e', 'E']):
                return int(float(value))
            return int(value)
        except (TypeError, ValueError):
            return int(default)
    
    def create_model(self) -> nn.Module:
        """モデルを作成"""
        model_name = self.config.get('model', {}).get('name', 'ResNet18')
        
        if model_name.lower() == 'resnet18':
            model = resnet18()
            # CIFAR-10用に最終層を調整
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"サポートされていないモデル: {model_name}")
        
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """オプティマイザーを作成"""
        training_config = self.config.get('training', {})
        optimizer_type = training_config.get('optimizer', 'Adam')
        lr = self._to_float(training_config.get('learning_rate', 0.1), 0.1)
        wd = self._to_float(training_config.get('weight_decay', 5e-4), 5e-4)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_type.lower() == 'sgd':
            momentum = self._to_float(training_config.get('momentum', 0.9), 0.9)
            return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        else:
            raise ValueError(f"サポートされていないオプティマイザー: {optimizer_type}")
    
    def create_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> Optional[object]:
        """スケジューラーを作成"""
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'StepLR')
        
        if scheduler_type == 'StepLR':
            step_size_ratio = self._to_float(scheduler_config.get('step_size_ratio', 0.4), 0.4)
            gamma = self._to_float(scheduler_config.get('gamma', 0.5), 0.5)
            step_size = int(self._to_int(num_epochs, num_epochs) * step_size_ratio)
            return StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"サポートされていないスケジューラー: {scheduler_type}")
    
    def create_swa_model(self, model: nn.Module) -> Tuple[Optional[object], Optional[object]]:
        """SWAモデルとスケジューラーを作成"""
        swa_config = self.config.get('swa', {})
        method = swa_config.get('method', 'normal')
        swa_lr = self._to_float(swa_config.get('swa_lr', 0.01), 0.01)
        
        if method == 'swa':
            swa_model = AveragedModel(model).to(self.device)
            swa_scheduler = SWALR(None, swa_lr=swa_lr)  # optimizerは後で設定
            return swa_model, swa_scheduler
        
        elif method == 'threshold_swa':
            threshold_config = self.config.get('threshold_swa', {})
            threshold = threshold_config.get('threshold', '0.0')
            mode = threshold_config.get('mode', 'gt')
            update_type = threshold_config.get('update_type', 'masking')
            selection_type = threshold_config.get('selection_type', 'threshold')
            topk_ratio = threshold_config.get('topk_ratio', 0.1)
            n_epochs = swa_config.get('interval', 1)
            
            swa_model = StepwiseConditionalSWA(
                model,
                threshold=threshold,
                mode=mode,
                update_type=update_type,
                selection_type=selection_type,
                topk_ratio=topk_ratio,
                n_epochs=n_epochs
            )
            swa_model.to(self.device)
            swa_scheduler = SWALR(None, swa_lr=swa_lr)  # optimizerは後で設定
            return swa_model, swa_scheduler
        
        else:
            return None, None
    
    def setup_swa_scheduler(self, swa_scheduler: Optional[object], optimizer: optim.Optimizer) -> None:
        """SWAスケジューラーにオプティマイザーを設定"""
        if swa_scheduler is not None:
            swa_scheduler.optimizer = optimizer


class TrainingComponents:
    """訓練に必要なコンポーネントをまとめて管理"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.factory = ModelFactory(config, device)
        
        # コンポーネントを作成
        self.model = self.factory.create_model()
        self.optimizer = self.factory.create_optimizer(self.model)
        self.scheduler = self.factory.create_scheduler(self.optimizer, self._get_num_epochs())
        self.swa_model, self.swa_scheduler = self.factory.create_swa_model(self.model)
        
        # SWAスケジューラーにオプティマイザーを設定
        self.factory.setup_swa_scheduler(self.swa_scheduler, self.optimizer)
        
        # 損失関数
        self.criterion = nn.CrossEntropyLoss()
    
    def _get_num_epochs(self) -> int:
        """エポック数を取得"""
        return self.config.get('training', {}).get('epochs', 100)
    
    def get_swa_method(self) -> str:
        """SWA方法を取得"""
        return self.config.get('swa', {}).get('method', 'normal')
    
    def get_swa_start_epoch(self) -> int:
        """SWA開始エポックを取得"""
        num_epochs = self._get_num_epochs()
        start_ratio = self.config.get('swa', {}).get('start_ratio', 0.75)
        return int(num_epochs * start_ratio)
    
    def get_swa_interval(self) -> int:
        """SWA更新間隔を取得"""
        return self.config.get('swa', {}).get('interval', 1)
