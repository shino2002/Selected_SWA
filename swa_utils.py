"""
SWA関連のユーティリティクラス
"""
import torch
from copy import deepcopy
from typing import Union, Optional, Dict, Any
from abc import ABC, abstractmethod


class ThresholdResolver:
    """動的しきい値解決クラス"""
    
    @staticmethod
    def resolve(threshold: Union[float, str], param: torch.Tensor) -> float:
        """しきい値を解決"""
        if isinstance(threshold, str):
            return ThresholdResolver._resolve_string_threshold(threshold, param)
        else:
            return float(threshold)
    
    @staticmethod
    def _resolve_string_threshold(threshold: str, param: torch.Tensor) -> float:
        """文字列形式のしきい値を解決"""
        flat = param.abs().view(-1)
        
        if threshold == 'mean':
            return flat.mean().item()
        elif threshold.startswith('mean+'):
            coeff = float(threshold.split('+')[1].replace('std', ''))
            return (flat.mean() + coeff * flat.std()).item()
        elif threshold.startswith('mean-'):
            coeff = float(threshold.split('-')[1].replace('std', ''))
            return (flat.mean() - coeff * flat.std()).item()
        elif threshold == 'median':
            return flat.median().item()
        elif threshold.startswith('percentile'):
            p = float(threshold.replace('percentile', ''))
            return torch.quantile(flat, p / 100.0).item()
        else:
            try:
                return float(threshold)
            except ValueError:
                raise ValueError(f"Unsupported threshold format: {threshold}")


class ParameterSelector(ABC):
    """パラメータ選択の抽象基底クラス"""
    
    @abstractmethod
    def create_mask(self, param: torch.Tensor) -> torch.Tensor:
        """マスクを作成"""
        pass


class ThresholdSelector(ParameterSelector):
    """しきい値ベースのパラメータ選択"""
    
    def __init__(self, threshold: Union[float, str], mode: str):
        self.threshold = threshold
        self.mode = mode
        assert mode in ['gt', 'lt'], f"Invalid mode: {mode}"
    
    def create_mask(self, param: torch.Tensor) -> torch.Tensor:
        """しきい値ベースのマスクを作成"""
        threshold_value = ThresholdResolver.resolve(self.threshold, param)
        
        if self.mode == 'gt':
            return param.abs() > threshold_value
        else:
            return param.abs() < threshold_value


class TopKSelector(ParameterSelector):
    """TopKベースのパラメータ選択"""
    
    def __init__(self, ratio: float, largest: bool = True):
        self.ratio = ratio
        self.largest = largest
    
    def create_mask(self, param: torch.Tensor) -> torch.Tensor:
        """TopKベースのマスクを作成"""
        abs_param = param.abs().flatten()
        k = int(self.ratio * abs_param.numel())
        
        if k == 0:
            return torch.zeros_like(param, dtype=torch.bool)
        
        _, topk_indices = torch.topk(abs_param, k=k, largest=self.largest)
        mask = torch.zeros_like(abs_param, dtype=torch.bool)
        mask[topk_indices] = True
        
        return mask.view_as(param)


class ParameterUpdater(ABC):
    """パラメータ更新の抽象基底クラス"""
    
    @abstractmethod
    def update(self, swa_param: torch.Tensor, model_param: torch.Tensor, 
               mask: torch.Tensor, update_count: int) -> None:
        """パラメータを更新"""
        pass


class MaskingUpdater(ParameterUpdater):
    """マスキングベースの更新"""
    
    def update(self, swa_param: torch.Tensor, model_param: torch.Tensor, 
               mask: torch.Tensor, update_count: int) -> None:
        """マスキングベースでパラメータを更新"""
        swa_param[mask] = (swa_param[mask] * update_count + model_param[mask]) / (update_count + 1)


class WeightedUpdater(ParameterUpdater):
    """重み付き更新"""
    
    def __init__(self, selector: ParameterSelector):
        self.selector = selector
        self.total_weights: Dict[str, torch.Tensor] = {}
    
    def update(self, swa_param: torch.Tensor, model_param: torch.Tensor, 
               mask: torch.Tensor, update_count: int) -> None:
        """重み付きでパラメータを更新"""
        # 重みを計算
        weight = self._calculate_weight(model_param, mask)
        
        # 累積重みを更新
        param_id = id(swa_param)
        if param_id not in self.total_weights:
            self.total_weights[param_id] = torch.zeros_like(swa_param)
        
        total_w = self.total_weights[param_id]
        swa_param.data = (swa_param.data * total_w + model_param * weight) / (total_w + weight + 1e-8)
        self.total_weights[param_id] = total_w + weight
    
    def _calculate_weight(self, param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """重みを計算"""
        if isinstance(self.selector, ThresholdSelector):
            return self._calculate_threshold_weight(param, mask)
        elif isinstance(self.selector, TopKSelector):
            return self._calculate_topk_weight(param, mask)
        else:
            raise ValueError(f"Unsupported selector type: {type(self.selector)}")
    
    def _calculate_threshold_weight(self, param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """しきい値ベースの重みを計算"""
        abs_param = param.abs()
        threshold_value = ThresholdResolver.resolve(self.selector.threshold, param)
        max_val = abs_param.max() + 1e-8
        
        if self.selector.mode == 'gt':
            raw_weight = abs_param - threshold_value
        else:
            raw_weight = threshold_value - abs_param
        
        return torch.clamp(raw_weight / (max_val - threshold_value + 1e-8), min=0.0)
    
    def _calculate_topk_weight(self, param: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """TopKベースの重みを計算"""
        flat_abs = param.abs().flatten()
        k = int(self.selector.ratio * flat_abs.numel())
        
        if k == 0:
            return torch.zeros_like(param)
        
        if self.selector.largest:
            threshold_value = torch.kthvalue(flat_abs, flat_abs.numel() - k).values
        else:
            threshold_value = torch.kthvalue(flat_abs, k).values
        
        max_val = flat_abs.max() + 1e-6
        norm_weight = (param.abs() - threshold_value) / (max_val - threshold_value + 1e-6)
        
        return torch.clamp(norm_weight, min=0.0)


class StepwiseConditionalSWA:
    """段階的条件付きSWA実装（リファクタリング版）"""
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        threshold: Union[float, str] = 0.0, 
        mode: str = 'gt',
        update_type: str = 'masking',
        selection_type: str = 'threshold', 
        topk_ratio: float = 0.1,
        n_steps: Optional[int] = None, 
        n_epochs: Optional[int] = None,
        debug: bool = True
    ):
        # バリデーション
        assert update_type in ['masking', 'weighted'], f"Invalid update_type: {update_type}"
        assert selection_type in ['threshold', 'topk', 'bottomk', 'none'], f"Invalid selection_type: {selection_type}"
        assert (n_steps is not None) ^ (n_epochs is not None), "Specify either n_steps or n_epochs"
        
        # 基本設定
        self.swa_model = deepcopy(model)
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.step_counter = 0
        self.update_count = 0
        self.debug = debug
        
        # セレクターとアップデーターを設定
        self.selector = self._create_selector(selection_type, threshold, mode, topk_ratio)
        self.updater = self._create_updater(update_type, self.selector)
    
    def _create_selector(self, selection_type: str, threshold: Union[float, str], 
                        mode: str, topk_ratio: float) -> ParameterSelector:
        """セレクターを作成"""
        if selection_type == 'threshold':
            return ThresholdSelector(threshold, mode)
        elif selection_type == 'topk':
            return TopKSelector(topk_ratio, largest=True)
        elif selection_type == 'bottomk':
            return TopKSelector(topk_ratio, largest=False)
        else:
            raise ValueError(f"Unknown selection_type: {selection_type}")
    
    def _create_updater(self, update_type: str, selector: ParameterSelector) -> ParameterUpdater:
        """アップデーターを作成"""
        if update_type == 'masking':
            return MaskingUpdater()
        elif update_type == 'weighted':
            return WeightedUpdater(selector)
        else:
            raise ValueError(f"Unknown update_type: {update_type}")
    
    def maybe_update(self, model: torch.nn.Module, current_step: Optional[int] = None, 
                    current_epoch: Optional[int] = None) -> None:
        """条件に応じてSWAモデルを更新"""
        # 更新タイミングをチェック
        if not self._should_update(current_step, current_epoch):
            return
        
        # 各パラメータを更新
        for (name, swa_param), (_, model_param) in zip(
            self.swa_model.named_parameters(), model.named_parameters()
        ):
            param_data = model_param.data
            swa_data = swa_param.data
            
            # マスクを作成
            mask = self.selector.create_mask(param_data)
            
            # デバッグ情報を出力
            if self.debug:
                mask_ratio = mask.sum().item() / mask.numel()
                print(f"[DEBUG] {name} mask ratio: {mask_ratio:.4f} | shape: {param_data.shape}")
            
            # パラメータを更新
            self.updater.update(swa_data, param_data, mask, self.update_count)
        
        self.update_count += 1
    
    def _should_update(self, current_step: Optional[int], current_epoch: Optional[int]) -> bool:
        """更新すべきかどうかを判定"""
        if self.n_steps is not None:
            self.step_counter += 1
            return self.step_counter % self.n_steps == 0
        elif self.n_epochs is not None:
            return current_epoch is not None and current_epoch % self.n_epochs == 0
        return False
    
    def apply_swa_weights(self, model: torch.nn.Module) -> None:
        """SWA重みをモデルに適用"""
        for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
            param.data.copy_(swa_param.data)
    
    def to(self, device: torch.device) -> None:
        """デバイスに移動"""
        self.swa_model.to(device)
    
    def state_dict(self) -> Dict[str, Any]:
        """状態辞書を取得"""
        return self.swa_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """状態辞書を読み込み"""
        self.swa_model.load_state_dict(state_dict)
