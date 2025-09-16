import yaml
import argparse
import os
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        設定辞書
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    コマンドライン引数で設定を上書きする
    
    Args:
        config: 設定辞書
        args: コマンドライン引数
        
    Returns:
        マージされた設定辞書
    """
    # コマンドライン引数で上書き可能な設定をマッピング
    arg_mappings = {
        'epochs': 'training.epochs',
        'swa': 'swa.start_ratio',
        'interval': 'swa.interval',
        'lr': 'training.learning_rate',
        'wd': 'training.weight_decay',
        'target_dir': 'experiment.output_dir',
        'model': 'model.name',
        'csv_name': 'experiment.csv_name',
        'method': 'swa.method',
        'mode': 'threshold_swa.mode',
        'threshold': 'threshold_swa.threshold',
        'selection_type': 'threshold_swa.selection_type',
        'update_type': 'threshold_swa.update_type',
        'topk_ratio': 'threshold_swa.topk_ratio',
    }
    
    for arg_name, config_path in arg_mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            # ネストされた辞書のキーを分割
            keys = config_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = getattr(args, arg_name)
    
    return config


def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    ネストされた辞書から値を取得する
    
    Args:
        config: 設定辞書
        key_path: ドット区切りのキーパス（例: "training.learning_rate"）
        default: デフォルト値
        
    Returns:
        設定値またはデフォルト値
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def create_config_parser() -> argparse.ArgumentParser:
    """
    設定ファイル対応のargparseパーサーを作成
    
    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Trains a classifier with YAML config support')
    
    # 設定ファイル関連
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='設定ファイルのパス (default: config.yaml)')
    
    # 基本設定（設定ファイルで上書き可能）
    parser.add_argument('--epochs', type=int, help='訓練エポック数')
    parser.add_argument('--swa', type=float, help='SWA開始比率')
    parser.add_argument('--interval', type=int, help='SWA更新間隔')
    parser.add_argument('--lr', type=float, help='学習率')
    parser.add_argument('--wd', type=float, help='重み減衰')
    
    parser.add_argument('--target_dir', type=str, help='出力ディレクトリ')
    parser.add_argument('--model', type=str, help='モデル名')
    parser.add_argument('--csv_name', type=str, help='CSVログファイル名')
    parser.add_argument('--method', choices=['normal', 'swa', 'threshold_swa'], 
                       help='訓練方法')
    parser.add_argument('--mode', choices=['gt', 'lt'], help='しきい値モード')
    
    parser.add_argument('--threshold', type=str, help='しきい値')
    parser.add_argument('--selection_type', choices=['threshold', 'topk', 'bottomk', 'none'], 
                       help='選択タイプ')
    parser.add_argument('--update_type', choices=['masking', 'weighted'], 
                       help='更新タイプ')
    parser.add_argument('--topk_ratio', type=float, help='TopK比率')
    
    return parser


def generate_experiment_name(config: Dict[str, Any]) -> str:
    """
    実験名を生成する
    
    Args:
        config: 設定辞書
        
    Returns:
        生成された実験名
    """
    # 現在の日時を取得
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    
    # 実験の基本情報を取得
    base_name = config.get('experiment', {}).get('name', 'experiment')
    method = config.get('swa', {}).get('method', 'normal')
    epochs = config.get('training', {}).get('epochs', 100)
    lr = config.get('training', {}).get('learning_rate', 0.1)
    
    # 実験名を構築
    experiment_parts = [date_str, base_name]
    
    # 方法を短縮形に変換
    method_short = {
        'normal': 'norm',
        'swa': 'swa',
        'threshold_swa': 'tswa'
    }.get(method, method)
    experiment_parts.append(method_short)
    
    # SWAの詳細情報を追加
    if method == 'threshold_swa':
        threshold = config.get('threshold_swa', {}).get('threshold', '0.0')
        selection_type = config.get('threshold_swa', {}).get('selection_type', 'threshold')
        update_type = config.get('threshold_swa', {}).get('update_type', 'masking')
        
        # しきい値を短縮形に変換
        if isinstance(threshold, str):
            if threshold.startswith('mean+'):
                coeff = threshold.split('+')[1].replace('std', '')
                threshold_short = f"m+{coeff}s"
            elif threshold.startswith('mean-'):
                coeff = threshold.split('-')[1].replace('std', '')
                threshold_short = f"m-{coeff}s"
            elif threshold.startswith('percentile'):
                p = threshold.replace('percentile', '')
                threshold_short = f"p{p}"
            else:
                threshold_short = threshold
        else:
            threshold_short = str(threshold)
        
        # 選択タイプと更新タイプを短縮形に変換
        selection_short = {
            'threshold': 'th',
            'topk': 'tk',
            'bottomk': 'bk'
        }.get(selection_type, selection_type)
        
        update_short = {
            'masking': 'mask',
            'weighted': 'wgt'
        }.get(update_type, update_type)
        
        experiment_parts.extend([f"t{threshold_short}", selection_short, update_short])
    
    # 学習率を追加（小数点以下1桁）
    lr_short = f"lr{lr:.1f}".replace('.0', '')
    experiment_parts.append(lr_short)
    
    # エポック数を追加
    experiment_parts.append(f"ep{epochs}")
    
    return "_".join(experiment_parts)


def setup_config() -> Dict[str, Any]:
    """
    設定ファイルとコマンドライン引数を統合して最終的な設定を返す
    
    Returns:
        統合された設定辞書
    """
    parser = create_config_parser()
    args = parser.parse_args()
    
    # 設定ファイルを読み込み
    config = load_config(args.config)
    
    # コマンドライン引数で上書き
    config = merge_config_with_args(config, args)
    
    # 実験名を生成して設定に追加
    experiment_name = generate_experiment_name(config)
    if 'experiment' not in config:
        config['experiment'] = {}
    config['experiment']['generated_name'] = experiment_name
    
    return config


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    設定を整形して表示する
    
    Args:
        config: 設定辞書
        indent: インデントレベル
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


if __name__ == "__main__":
    # テスト用
    config = setup_config()
    print("=== 設定内容 ===")
    print_config(config)
