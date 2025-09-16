"""
メインの訓練スクリプト（リファクタリング版）
"""
import torch
import sys
import os

# パスを追加
sys.path.append('..')

from config_loader import setup_config, get_nested_value
from data_utils import DataManager
from trainer import Trainer


def setup_device(config: dict) -> torch.device:
    """デバイスを設定"""
    device_config = get_nested_value(config, 'experiment.device', 'auto')
    
    if device_config == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_config)


def print_config_summary(config: dict) -> None:
    """設定の概要を表示"""
    print("=== 実験設定 ===")
    print(f"実験名: {get_nested_value(config, 'experiment.name', 'unnamed')}")
    print(f"デバイス: {get_nested_value(config, 'experiment.device', 'auto')}")
    print(f"エポック数: {get_nested_value(config, 'training.epochs', 100)}")
    print(f"学習率: {get_nested_value(config, 'training.learning_rate', 0.1)}")
    print(f"SWA方法: {get_nested_value(config, 'swa.method', 'normal')}")
    
    if get_nested_value(config, 'swa.method', 'normal') == 'threshold_swa':
        print(f"しきい値: {get_nested_value(config, 'threshold_swa.threshold', '0.0')}")
        print(f"選択タイプ: {get_nested_value(config, 'threshold_swa.selection_type', 'threshold')}")
        print(f"更新タイプ: {get_nested_value(config, 'threshold_swa.update_type', 'masking')}")
    
    print("=" * 50)


def main():
    """メイン関数"""
    # 設定を読み込み
    config = setup_config()
    
    # デバイスを設定
    device = setup_device(config)
    
    # 設定概要を表示
    print_config_summary(config)
    
    # データマネージャーを作成
    data_manager = DataManager(config)
    trainloader, testloader = data_manager.get_dataloaders()
    
    print(f"データセット: {data_manager.num_classes}クラス")
    print(f"訓練データ: {len(trainloader.dataset)}サンプル")
    print(f"テストデータ: {len(testloader.dataset)}サンプル")
    print()
    
    # トレーナーを作成して訓練を実行
    trainer = Trainer(config, device)
    trainer.train(trainloader, testloader)
    
    print("訓練が完了しました！")


if __name__ == "__main__":
    main()
# テスト用のコメント
