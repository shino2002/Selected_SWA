import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import argparse

def plot_multi_logs(log_paths, labels=None):
    if labels is None:
        labels = log_paths

    # フルパス化と事前ロード
    data = []
    for path, label in zip(log_paths, labels):
        full_path = os.path.join("./log", path)
        df = pd.read_csv(full_path)
        data.append((df, label))

    colors = cm.tab10.colors  # 最大10色対応
    # import pdb; pdb.set_trace()

    plt.clf()
    # ---- LOSS CURVE ----
    plt.figure(figsize=(10, 5))
    for idx, (df, label) in enumerate(data):
        color = colors[idx % len(colors)]
        plt.plot(df['epoch'], df['train_loss'], label=f'{label} - train', linestyle='-', color=color)
        plt.plot(df['epoch'], df['test_loss'],  label=f'{label} - test',  linestyle='--', color=color)
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_weighted2.png')
    plt.show()

    plt.clf()
    # ---- ACCURACY CURVE ----
    plt.figure(figsize=(10, 5))
    for idx, (df, label) in enumerate(data):
        color = colors[idx % len(colors)]
        plt.plot(df['epoch'], df['train_acc'], label=f'{label} - train', linestyle='-', color=color)
        plt.plot(df['epoch'], df['test_acc'],  label=f'{label} - test',  linestyle='--', color=color)
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('accuracy_curve_weighted2.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', required=True, help='List of CSV log files in ./log/')
    parser.add_argument('--labels', nargs='+', help='List of labels corresponding to logs (optional)')
    args = parser.parse_args()

    # 長さが合わない場合やラベル未指定時はファイル名をそのまま使う
    if not args.labels or len(args.labels) != len(args.logs):
        labels = [os.path.splitext(os.path.basename(p))[0] for p in args.logs]
    else:
        labels = args.labels

    plot_multi_logs(args.logs, labels)


