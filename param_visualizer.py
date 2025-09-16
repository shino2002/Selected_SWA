import os
import torch
import matplotlib.pyplot as plt

class ParamHistogramVisualizer:
    def __init__(self, output_dir="param_hist", every_n_epoch=10, selected_layers=None):
        self.output_dir = output_dir
        self.every_n_epoch = every_n_epoch
        self.selected_layers = selected_layers  # List of layer names or None
        os.makedirs(output_dir, exist_ok=True)

    def maybe_plot(self, model, epoch):
        # if epoch % self.every_n_epoch != 0:
        #     return

        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if param.dim() == 0:  # skip scalar params
                continue
            if self.selected_layers is not None and name not in self.selected_layers:
                continue

            data = param.view(-1).cpu().numpy()
            plt.figure(figsize=(6, 3))
            plt.hist(data, bins=100, alpha=0.7, density=True)
            plt.title(f"{name} (epoch {epoch})")
            plt.xlabel("weight value")
            plt.ylabel("density")
            plt.grid(True)
            filename = f"{name.replace('.', '_')}_epoch{epoch+1}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
