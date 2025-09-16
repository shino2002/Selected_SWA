"""
実験結果を管理するクラス
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import shutil


class ExperimentManager:
    """実験結果を管理するクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = self._generate_experiment_name()
        self.experiment_dir = self._create_experiment_directory()
        self._save_config()
    
    def _generate_experiment_name(self) -> str:
        """実験名を生成"""
        # 現在の日時を取得
        now = datetime.now()
        date_str = now.strftime("%Y%m%d_%H%M%S")
        
        # 実験の基本情報を取得
        base_name = self.config.get('experiment', {}).get('name', 'experiment')
        method = self.config.get('swa', {}).get('method', 'normal')
        epochs = self.config.get('training', {}).get('epochs', 100)
        lr = self.config.get('training', {}).get('learning_rate', 0.1)
        
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
            threshold = self.config.get('threshold_swa', {}).get('threshold', '0.0')
            selection_type = self.config.get('threshold_swa', {}).get('selection_type', 'threshold')
            update_type = self.config.get('threshold_swa', {}).get('update_type', 'masking')
            
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
    
    def _create_experiment_directory(self) -> str:
        """実験ディレクトリを作成"""
        base_output_dir = self.config.get('experiment', {}).get('output_dir', 'out')
        experiment_path = os.path.join(base_output_dir, self.experiment_name)
        
        # ディレクトリを作成
        os.makedirs(experiment_path, exist_ok=True)
        
        # サブディレクトリを作成
        subdirs = ['logs', 'plots', 'models', 'configs']
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_path, subdir), exist_ok=True)
        
        return experiment_path
    
    def _save_config(self) -> None:
        """設定ファイルを実験ディレクトリに保存"""
        config_path = os.path.join(self.experiment_dir, 'configs', 'config.yaml')
        
        # YAMLファイルとして保存
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # JSONファイルとしても保存（メタデータ付き）
        config_with_meta = {
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'config': self.config
        }
        
        json_path = os.path.join(self.experiment_dir, 'configs', 'config.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
    
    def get_log_path(self, filename: str = 'log.csv') -> str:
        """ログファイルのパスを取得"""
        return os.path.join(self.experiment_dir, 'logs', filename)
    
    def get_plot_path(self, filename: str) -> str:
        """プロットファイルのパスを取得"""
        return os.path.join(self.experiment_dir, 'plots', filename)
    
    def get_model_path(self, filename: str) -> str:
        """モデルファイルのパスを取得"""
        return os.path.join(self.experiment_dir, 'models', filename)
    
    def save_experiment_info(self, info: Dict[str, Any]) -> None:
        """実験情報を保存"""
        info_path = os.path.join(self.experiment_dir, 'experiment_info.json')
        
        experiment_info = {
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'config': self.config,
            'results': info
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_info, f, indent=2, ensure_ascii=False)
    
    def copy_plots(self, source_dir: str) -> None:
        """既存のプロットファイルを実験ディレクトリにコピー"""
        plots_dir = os.path.join(self.experiment_dir, 'plots')
        
        if os.path.exists(source_dir):
            for filename in os.listdir(source_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    src_path = os.path.join(source_dir, filename)
                    dst_path = os.path.join(plots_dir, filename)
                    shutil.copy2(src_path, dst_path)
    
    def get_experiment_summary(self) -> str:
        """実験の概要を文字列で取得"""
        method = self.config.get('swa', {}).get('method', 'normal')
        epochs = self.config.get('training', {}).get('epochs', 100)
        lr = self.config.get('training', {}).get('learning_rate', 0.1)
        
        summary = f"実験名: {self.experiment_name}\n"
        summary += f"方法: {method}\n"
        summary += f"エポック数: {epochs}\n"
        summary += f"学習率: {lr}\n"
        
        if method == 'threshold_swa':
            threshold = self.config.get('threshold_swa', {}).get('threshold', '0.0')
            selection_type = self.config.get('threshold_swa', {}).get('selection_type', 'threshold')
            update_type = self.config.get('threshold_swa', {}).get('update_type', 'masking')
            
            summary += f"しきい値: {threshold}\n"
            summary += f"選択タイプ: {selection_type}\n"
            summary += f"更新タイプ: {update_type}\n"
        
        summary += f"結果ディレクトリ: {self.experiment_dir}\n"
        
        return summary
    
    def create_readme(self) -> None:
        """実験ディレクトリにREADMEファイルを作成"""
        readme_path = os.path.join(self.experiment_dir, 'README.md')
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# 実験結果: {self.experiment_name}\n\n")
            f.write(f"## 実験概要\n\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            
            # 実験の詳細情報を追加
            method = self.config.get('swa', {}).get('method', 'normal')
            epochs = self.config.get('training', {}).get('epochs', 100)
            lr = self.config.get('training', {}).get('learning_rate', 0.1)
            
            f.write(f"**実験設定:**\n")
            f.write(f"- 方法: {method}\n")
            f.write(f"- エポック数: {epochs}\n")
            f.write(f"- 学習率: {lr}\n")
            
            if method == 'threshold_swa':
                threshold_config = self.config.get('threshold_swa', {})
                f.write(f"- しきい値: {threshold_config.get('threshold', '0.0')}\n")
                f.write(f"- 選択タイプ: {threshold_config.get('selection_type', 'threshold')}\n")
                f.write(f"- 更新タイプ: {threshold_config.get('update_type', 'masking')}\n")
            
            f.write(f"\n## 設定\n\n")
            f.write(f"```yaml\n")
            import yaml
            f.write(yaml.dump(self.config, default_flow_style=False, allow_unicode=True))
            f.write(f"```\n\n")
            
            f.write(f"## ディレクトリ構造\n\n")
            f.write(f"- `logs/`: ログファイル（CSV形式）\n")
            f.write(f"- `plots/`: グラフ画像\n")
            f.write(f"  - `loss_curves.png`: 損失曲線\n")
            f.write(f"  - `accuracy_curves.png`: 精度曲線\n")
            f.write(f"  - `learning_curves.png`: 学習曲線（損失・精度）\n")
            f.write(f"  - `swa_analysis.png`: SWA分析（SWA使用時）\n")
            f.write(f"  - `experiment_summary.png`: 実験概要\n")
            f.write(f"- `models/`: 保存されたモデル\n")
            f.write(f"- `configs/`: 設定ファイル\n")
            
            f.write(f"\n## 結果ファイル\n\n")
            f.write(f"- **ログファイル**: `logs/log.csv` - 各エポックの損失と精度\n")
            f.write(f"- **モデルファイル**: `models/model_実験名.pth` - 最終的なモデルの重み\n")
            f.write(f"- **設定ファイル**: `configs/config.yaml` - 実験で使用した設定\n")
            f.write(f"- **実験情報**: `experiment_info.json` - 実験の詳細情報\n")


def create_experiment_manager(config: Dict[str, Any]) -> ExperimentManager:
    """実験マネージャーを作成するファクトリー関数"""
    return ExperimentManager(config)
