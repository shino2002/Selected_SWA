"""
訓練ロジックを管理するTrainerクラス
"""
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional

from logger import TrainingLogger
from data_utils import update_bn_on_device
from model_factory import TrainingComponents
from experiment_manager import ExperimentManager
from plot_utils import create_learning_curves


class Trainer:
    """訓練を管理するクラス"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.components = TrainingComponents(config, device)
        self.experiment_manager = ExperimentManager(config)
        self.logger = self._setup_logger()
        self.step = 0
    
    def _setup_logger(self) -> TrainingLogger:
        """ロガーを設定"""
        csv_name = self.config.get('experiment', {}).get('csv_name', 'log.csv')
        filepath = self.experiment_manager.get_log_path(csv_name)
        
        return TrainingLogger(filepath)
    
    def train_epoch(self, trainloader: DataLoader) -> Tuple[float, float]:
        """1エポックの訓練を実行"""
        self.components.model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0
        
        for x, y in trainloader:
            x, y = x.to(self.device), y.to(self.device)
            
            # 順伝播
            self.components.optimizer.zero_grad()
            out = self.components.model(x)
            loss = self.components.criterion(out, y)
            
            # 逆伝播
            loss.backward()
            self.components.optimizer.step()
            
            # 統計更新
            pred = out.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_correct += (pred == y).sum().item()
            total_count += y.size(0)
            self.step += 1
        
        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        
        return train_loss, train_acc
    
    def evaluate(self, trainloader: DataLoader, testloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """モデルを評価"""
        method = self.components.get_swa_method()
        swa_start = self.components.get_swa_start_epoch()
        
        # 評価用モデルを準備
        if method == 'swa' and epoch >= swa_start:
            model_for_eval = deepcopy(self.components.model).to(self.device)
            self.components.swa_model.copy_to(model_for_eval)
            update_bn_on_device(trainloader, model_for_eval, self.device)
        elif method == 'threshold_swa' and epoch >= swa_start:
            model_for_eval = deepcopy(self.components.model).to(self.device)
            self.components.swa_model.apply_swa_weights(model_for_eval)
            update_bn_on_device(trainloader, model_for_eval, self.device)
        else:
            model_for_eval = self.components.model
        
        # 評価実行
        model_for_eval.eval()
        total_loss, total_correct, total_count = 0.0, 0, 0
        
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(self.device), y.to(self.device)
                out = model_for_eval(x)
                loss = self.components.criterion(out, y)
                
                pred = out.argmax(dim=1)
                total_loss += loss.item() * y.size(0)
                total_correct += (pred == y).sum().item()
                total_count += y.size(0)
        
        test_loss = total_loss / total_count
        test_acc = total_correct / total_count
        
        return test_loss, test_acc
    
    def update_swa(self, epoch: int) -> None:
        """SWAモデルを更新"""
        method = self.components.get_swa_method()
        swa_start = self.components.get_swa_start_epoch()
        swa_interval = self.components.get_swa_interval()
        
        if method == 'swa' and epoch >= swa_start:
            self.components.swa_model.update_parameters(self.components.model)
            self.components.swa_scheduler.step()
        elif method == 'threshold_swa' and epoch >= swa_start:
            self.components.swa_model.maybe_update(self.components.model, current_epoch=epoch)
            self.components.swa_scheduler.step()
        elif method in ['swa', 'normal', 'threshold_swa']:
            self.components.scheduler.step()
    
    def train(self, trainloader: DataLoader, testloader: DataLoader) -> None:
        """全体の訓練を実行"""
        num_epochs = self.config.get('training', {}).get('epochs', 100)
        
        # 実験情報を表示
        print(self.experiment_manager.get_experiment_summary())
        print()
        
        for epoch in tqdm(range(num_epochs), desc="Epochs", unit="epoch"):
            # 訓練
            train_loss, train_acc = self.train_epoch(trainloader)
            
            # SWA更新
            self.update_swa(epoch)
            
            # 評価
            test_loss, test_acc = self.evaluate(trainloader, testloader, epoch)
            
            # ログ記録
            self.logger.log_epoch(
                epoch=epoch, 
                step=self.step,
                train_loss=train_loss, 
                train_acc=train_acc,
                test_loss=test_loss, 
                test_acc=test_acc
            )
        
        # 実験完了後の処理
        self._finalize_experiment()
    
    def _finalize_experiment(self) -> None:
        """実験完了後の処理"""
        # モデルを保存
        model_filename = f"model_{self.experiment_manager.experiment_name}.pth"
        model_path = self.experiment_manager.get_model_path(model_filename)
        self.save_model(model_path)
        
        # 学習曲線をプロット
        csv_path = self.experiment_manager.get_log_path()
        swa_start_epoch = self.components.get_swa_start_epoch() if self.components.get_swa_method() != 'normal' else None
        
        print("学習曲線を生成中...")
        create_learning_curves(
            csv_path=csv_path,
            experiment_name=self.experiment_manager.experiment_name,
            output_dir=self.experiment_manager.experiment_dir,
            config=self.config,
            swa_start_epoch=swa_start_epoch
        )
        
        # 実験情報を保存
        experiment_info = {
            'final_step': self.step,
            'method': self.components.get_swa_method(),
            'model_path': model_path,
            'csv_path': csv_path,
            'plots_dir': os.path.join(self.experiment_manager.experiment_dir, 'plots')
        }
        self.experiment_manager.save_experiment_info(experiment_info)
        
        # READMEファイルを作成
        self.experiment_manager.create_readme()
        
        print(f"\n実験が完了しました！")
        print(f"結果は以下のディレクトリに保存されました: {self.experiment_manager.experiment_dir}")
        print(f"  - ログファイル: {csv_path}")
        print(f"  - モデルファイル: {model_path}")
        print(f"  - プロットファイル: {os.path.join(self.experiment_manager.experiment_dir, 'plots')}")
    
    def save_model(self, filepath: str) -> None:
        """モデルを保存"""
        method = self.components.get_swa_method()
        
        if method in ['swa', 'threshold_swa'] and self.components.swa_model is not None:
            torch.save(self.components.swa_model.state_dict(), filepath)
        else:
            torch.save(self.components.model.state_dict(), filepath)
    
    def load_model(self, filepath: str) -> None:
        """モデルを読み込み"""
        method = self.components.get_swa_method()
        
        if method in ['swa', 'threshold_swa'] and self.components.swa_model is not None:
            self.components.swa_model.load_state_dict(torch.load(filepath))
        else:
            self.components.model.load_state_dict(torch.load(filepath))
