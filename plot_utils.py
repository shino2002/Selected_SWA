"""
学習曲線をプロットするユーティリティ
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Dict, Any, Optional
import numpy as np


class LearningCurvePlotter:
    """学習曲線をプロットするクラス"""
    
    def __init__(self, experiment_name: str, output_dir: str):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 日本語フォントの設定
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_learning_curves(self, csv_path: str) -> None:
        """CSVファイルから学習曲線をプロット"""
        if not os.path.exists(csv_path):
            print(f"警告: CSVファイルが見つかりません: {csv_path}")
            return
        
        # CSVファイルを読み込み
        df = pd.read_csv(csv_path)
        # 列を数値化（非数値はNaNに）し、NaNを前方埋め→後方埋めで補完
        for col in ['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # epochがNaNの行は除去
        if 'epoch' in df.columns:
            df = df.dropna(subset=['epoch'])
            df['epoch'] = df['epoch'].astype(int)
        # 欠損補完
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 学習曲線をプロット
        self._plot_loss_curves(df)
        self._plot_accuracy_curves(df)
        self._plot_combined_curves(df)
    
    def _plot_loss_curves(self, df: pd.DataFrame) -> None:
        """損失曲線をプロット"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(df['epoch'], df['test_loss'], label='Test Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves - {self.experiment_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'loss_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"損失曲線を保存しました: {save_path}")
    
    def _plot_accuracy_curves(self, df: pd.DataFrame) -> None:
        """精度曲線をプロット"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', linewidth=2)
        plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Curves - {self.experiment_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'accuracy_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"精度曲線を保存しました: {save_path}")
    
    def _plot_combined_curves(self, df: pd.DataFrame) -> None:
        """損失と精度を組み合わせたプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 損失曲線
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(df['epoch'], df['test_loss'], label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 精度曲線
        ax2.plot(df['epoch'], df['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(df['epoch'], df['test_acc'], label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Learning Curves - {self.experiment_name}', fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'learning_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"学習曲線を保存しました: {save_path}")
    
    def plot_swa_comparison(self, csv_path: str, swa_start_epoch: int) -> None:
        """SWA開始前後の比較プロット"""
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(12, 8))
        
        # サブプロットを作成
        plt.subplot(2, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(df['epoch'], df['test_loss'], label='Test Loss', linewidth=2)
        plt.axvline(x=swa_start_epoch, color='red', linestyle='--', alpha=0.7, label='SWA Start')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', linewidth=2)
        plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', linewidth=2)
        plt.axvline(x=swa_start_epoch, color='red', linestyle='--', alpha=0.7, label='SWA Start')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # SWA開始前後の統計
        plt.subplot(2, 2, 3)
        before_swa = df[df['epoch'] < swa_start_epoch]
        after_swa = df[df['epoch'] >= swa_start_epoch]
        
        if len(before_swa) > 0 and len(after_swa) > 0:
            categories = ['Before SWA', 'After SWA']
            train_acc = [before_swa['train_acc'].mean(), after_swa['train_acc'].mean()]
            test_acc = [before_swa['test_acc'].mean(), after_swa['test_acc'].mean()]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.bar(x - width/2, train_acc, width, label='Train Accuracy', alpha=0.8)
            plt.bar(x + width/2, test_acc, width, label='Test Accuracy', alpha=0.8)
            
            plt.xlabel('Phase')
            plt.ylabel('Average Accuracy')
            plt.title('Accuracy Before/After SWA')
            plt.xticks(x, categories)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 最終結果の比較
        plt.subplot(2, 2, 4)
        final_epoch = df['epoch'].max()
        final_results = df[df['epoch'] == final_epoch]
        
        if len(final_results) > 0:
            metrics = ['Train Loss', 'Test Loss', 'Train Acc', 'Test Acc']
            values = [
                final_results['train_loss'].iloc[0],
                final_results['test_loss'].iloc[0],
                final_results['train_acc'].iloc[0],
                final_results['test_acc'].iloc[0]
            ]
            
            plt.bar(metrics, values, alpha=0.8)
            plt.title('Final Results')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 値をバーの上に表示
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'SWA Analysis - {self.experiment_name}', fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'swa_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SWA分析プロットを保存しました: {save_path}")
    
    def create_summary_plot(self, csv_path: str, config: Dict[str, Any]) -> None:
        """実験の概要プロットを作成"""
        if not os.path.exists(csv_path):
            return
        
        df = pd.read_csv(csv_path)
        
        fig = plt.figure(figsize=(16, 10))
        
        # メインの学習曲線
        plt.subplot(2, 3, 1)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(df['epoch'], df['test_loss'], label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', linewidth=2)
        plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 設定情報を表示
        plt.subplot(2, 3, 3)
        plt.axis('off')
        
        # 設定情報をテキストで表示
        config_text = f"""
Experiment: {self.experiment_name}

Method: {config.get('swa', {}).get('method', 'normal')}
Epochs: {config.get('training', {}).get('epochs', 100)}
Learning Rate: {config.get('training', {}).get('learning_rate', 0.1)}
Batch Size: {config.get('dataset', {}).get('batch_size', 64)}

Final Results:
Train Loss: {df['train_loss'].iloc[-1]:.4f}
Test Loss: {df['test_loss'].iloc[-1]:.4f}
Train Acc: {df['train_acc'].iloc[-1]:.4f}
Test Acc: {df['test_acc'].iloc[-1]:.4f}
        """
        
        if config.get('swa', {}).get('method') == 'threshold_swa':
            threshold_config = config.get('threshold_swa', {})
            config_text += f"""
SWA Details:
Threshold: {threshold_config.get('threshold', '0.0')}
Selection: {threshold_config.get('selection_type', 'threshold')}
Update: {threshold_config.get('update_type', 'masking')}
            """
        
        plt.text(0.1, 0.9, config_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 損失の分布
        plt.subplot(2, 3, 4)
        plt.hist(df['train_loss'], bins=20, alpha=0.7, label='Train Loss', density=True)
        plt.hist(df['test_loss'], bins=20, alpha=0.7, label='Test Loss', density=True)
        plt.xlabel('Loss')
        plt.ylabel('Density')
        plt.title('Loss Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 精度の分布
        plt.subplot(2, 3, 5)
        plt.hist(df['train_acc'], bins=20, alpha=0.7, label='Train Accuracy', density=True)
        plt.hist(df['test_acc'], bins=20, alpha=0.7, label='Test Accuracy', density=True)
        plt.xlabel('Accuracy')
        plt.ylabel('Density')
        plt.title('Accuracy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学習率の変化（もしあれば）
        plt.subplot(2, 3, 6)
        if 'lr' in df.columns:
            plt.plot(df['epoch'], df['lr'], linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Experiment Summary - {self.experiment_name}', fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.plots_dir, 'experiment_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"実験概要プロットを保存しました: {save_path}")


def create_learning_curves(csv_path: str, experiment_name: str, output_dir: str, 
                          config: Dict[str, Any], swa_start_epoch: Optional[int] = None) -> None:
    """学習曲線を作成する便利関数"""
    plotter = LearningCurvePlotter(experiment_name, output_dir)
    
    # 基本的な学習曲線
    plotter.plot_learning_curves(csv_path)
    
    # SWA分析（SWAを使用している場合）
    if swa_start_epoch is not None:
        plotter.plot_swa_comparison(csv_path, swa_start_epoch)
    
    # 実験概要
    plotter.create_summary_plot(csv_path, config)
