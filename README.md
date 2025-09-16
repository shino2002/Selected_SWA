# Selected SWA: Stepwise Conditional Stochastic Weight Averaging

このプロジェクトでは、Stepwise Conditional SWA（段階的条件付きSWA）の実験を行います。実験パラメータをYAMLファイルで管理できるようになり、パラメータ調整が簡単になり、実験の再現性が向上します。

## 概要

Selected SWAは、従来のStochastic Weight Averaging (SWA)を拡張した手法で、以下の特徴があります：

- **段階的条件付きSWA**: 特定の条件に基づいてモデルパラメータを選択的に平均化
- **しきい値ベース選択**: パラメータの値に基づいた選択基準
- **TopK選択**: 上位K個のパラメータのみを平均化
- **重み付き平均**: パラメータの重要度に基づいた重み付き平均
- **マスキング**: 選択されたパラメータのみを更新

## ディレクトリ構造

```
selected_swa/
├── train.py                    # メインの訓練スクリプト
├── main.py                     # 代替メインスクリプト
├── config.yaml                 # デフォルト設定ファイル
├── config_loader.py            # 設定ファイル読み込みユーティリティ
├── data_utils.py               # データセット管理クラス
├── model_factory.py            # モデル・オプティマイザー・スケジューラー作成
├── trainer.py                  # 訓練ロジック管理クラス
├── swa_utils.py                # SWA関連ユーティリティ
├── experiment_manager.py       # 実験結果管理クラス
├── plot_utils.py               # 学習曲線プロットユーティリティ
├── logger.py                   # ログ記録ユーティリティ
├── param_visualizer.py         # パラメータ可視化ユーティリティ
├── requirements.txt            # 依存関係
├── configs/                    # 設定ファイル例
│   ├── normal_training.yaml
│   ├── standard_swa.yaml
│   ├── threshold_swa_masking.yaml
│   ├── threshold_swa_weighted.yaml
│   └── topk_swa.yaml
├── scripts/                    # 実行スクリプト
│   ├── train.sh
│   ├── train_base.sh
│   ├── run_weighted.sh
│   └── plot_log.py
├── legacy/                     # 古いファイル
│   ├── Cond_SWA_debug.py
│   └── swa_conditioned.py
└── out/                        # 実験結果（自動生成）
    └── [実験フォルダ]/
        ├── logs/
        ├── plots/
        ├── models/
        └── configs/
```

## インストール

1. リポジトリをクローン：
```bash
git clone https://github.com/[username]/selected_swa.git
cd selected_swa
```

2. 依存関係をインストール：
```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法

```bash
# デフォルトのconfig.yamlを使用
python train.py

# 特定の設定ファイルを使用
python train.py --config configs/normal_training.yaml

# コマンドライン引数で設定を上書き
python train.py --config configs/standard_swa.yaml --epochs 150 --lr 0.05
```

### 利用可能な設定ファイル例

1. **`configs/normal_training.yaml`**: 通常の訓練
2. **`configs/standard_swa.yaml`**: 標準SWA
3. **`configs/threshold_swa_masking.yaml`**: しきい値SWA（マスキング）
4. **`configs/threshold_swa_weighted.yaml`**: しきい値SWA（重み付き）
5. **`configs/topk_swa.yaml`**: TopK選択SWA

### しきい値の設定方法

`threshold_swa.threshold`では以下の形式がサポートされています：

- 数値: `0.01`（固定値）
- `mean`: 平均値
- `mean+1std`: 平均+1標準偏差
- `mean-1std`: 平均-1標準偏差
- `median`: 中央値
- `percentile90`: 90パーセンタイル

### コマンドライン引数での上書き

設定ファイルの値はコマンドライン引数で上書きできます：

```bash
# 学習率を変更
python train.py --lr 0.05

# SWA方法を変更
python train.py --method threshold_swa --threshold "mean+1std"

# エポック数を変更
python train.py --epochs 200
```

## 実験結果管理機能

### 自動的な実験フォルダ作成

実験を実行すると、日付・時間・実験詳細を含む名前のフォルダが自動的に作成されます。

#### 実験名の命名規則

```
YYYYMMDD_HHMMSS_実験名_方法_詳細パラメータ_学習率_エポック数
```

**例:**
- `20250910_152954_cifar10_test_norm_lr0.1_ep100` (通常訓練)
- `20250910_152954_cifar10_swa_swa_lr0.1_ep200` (標準SWA)
- `20250910_152954_cifar10_threshold_tswa_tm+1s_th_mask_lr0.1_ep150` (しきい値SWA)

### 自動生成されるプロット

- **`loss_curves.png`**: 訓練・テスト損失の推移
- **`accuracy_curves.png`**: 訓練・テスト精度の推移
- **`learning_curves.png`**: 損失と精度を並べて表示
- **`swa_analysis.png`**: SWA開始前後の比較分析（SWA使用時のみ）
- **`experiment_summary.png`**: 実験の概要と統計情報

## アーキテクチャ

### 主要なクラス

1. **`DataManager`** (`data_utils.py`): データセットとトランスフォームを管理
2. **`ModelFactory`** (`model_factory.py`): モデル、オプティマイザー、スケジューラーを作成
3. **`TrainingComponents`** (`model_factory.py`): 訓練に必要なコンポーネントを統合管理
4. **`Trainer`** (`trainer.py`): 訓練ロジックを管理
5. **`StepwiseConditionalSWA`** (`swa_utils.py`): リファクタリングされたSWA実装
6. **`ExperimentManager`** (`experiment_manager.py`): 実験結果を管理
7. **`LearningCurvePlotter`** (`plot_utils.py`): 学習曲線をプロット

### 設計の利点

- **単一責任の原則**: 各クラスが明確な責任を持つ
- **依存性注入**: 設定を外部から注入して柔軟性を向上
- **テスタビリティ**: 各コンポーネントを独立してテスト可能
- **拡張性**: 新しいモデルやデータセットを簡単に追加可能
- **保守性**: コードの変更が局所化され、影響範囲が明確

## デバッグ機能

デバッグ情報を有効にするには、設定ファイルで以下を設定：

```yaml
debug:
  enabled: true
  print_mask_ratios: true  # マスク比率を表示
  print_weight_stats: true  # 重み統計を表示
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 詳細な設定情報

詳細な設定情報については、[README_config.md](README_config.md)を参照してください。
