import csv
import os

class TrainingLogger:
    def __init__(self, filepath='training_log.csv'):
        self.filepath = filepath
        self._initialized = False

    def log_epoch(self, epoch, step, train_loss, train_acc, test_loss, test_acc):
        if not self._initialized:
            self._init_file()

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }

        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def _init_file(self):
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        self._initialized = True
