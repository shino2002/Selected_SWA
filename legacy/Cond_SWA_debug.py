import torch
from copy import deepcopy

class StepwiseConditionalSWA:
    def __init__(self, model, threshold=0.0, mode='gt',
                 update_type='masking',
                 selection_type='threshold', topk_ratio=0.1,
                 n_steps=None, n_epochs=None,
                 debug=False):
        assert update_type in ['masking', 'weighted']
        assert selection_type in ['threshold', 'topk', 'bottomk', 'none']
        if selection_type == 'threshold':
            assert mode in ['gt', 'lt']
        assert (n_steps is not None) ^ (n_epochs is not None), "Specify either n_steps or n_epochs"

        self.swa_model = deepcopy(model)
        self.threshold = threshold
        self.mode = mode
        self.update_type = update_type
        self.selection_type = selection_type
        self.topk_ratio = topk_ratio
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.step_counter = 0
        self.n = 0
        self.debug = debug

        # weighted 用
        self.total_weights = {
            name: torch.zeros_like(p.data)
            for name, p in self.swa_model.named_parameters()
        }

    def maybe_update(self, model, current_step=None, current_epoch=None):
        if self.n_steps is not None:
            self.step_counter += 1
            if self.step_counter % self.n_steps != 0:
                return
        elif self.n_epochs is not None:
            if current_epoch is None or (current_epoch % self.n_epochs != 0):
                return

        for (name, swa_param), (_, model_param) in zip(
            self.swa_model.named_parameters(), model.named_parameters()
        ):
            param_data = model_param.data
            swa_data = swa_param.data

            # === maskの決定 ===
            mask = None
            if self.selection_type == 'threshold':
                if self.mode == 'gt':
                    mask = param_data.abs() > float(self.threshold)
                else:
                    mask = param_data.abs() < float(self.threshold)

            elif self.selection_type in ['topk', 'bottomk']:
                flat = param_data.abs().flatten()
                k = int(self.topk_ratio * flat.numel())
                if k == 0:
                    continue
                values, indices = torch.topk(flat, k=k, largest=(self.selection_type == 'topk'))
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask[indices] = True
                mask = mask.view_as(param_data)

            elif self.selection_type == 'none':
                mask = None  # 全体平均用

            else:
                raise ValueError(f"Unknown selection_type: {self.selection_type}")

            # === 平均更新 ===
            if self.update_type == 'masking':
                if mask is None:
                    swa_param.data = (swa_param.data * self.n + param_data) / (self.n + 1)
                    if self.debug:
                        print(f"[DEBUG] {name}: global avg (mask=None)")
                else:
                    swa_data[mask] = (swa_data[mask] * self.n + param_data[mask]) / (self.n + 1)
                    if self.debug:
                        ratio = mask.sum().item() / mask.numel()
                        print(f"[DEBUG] {name}: masked avg | ratio={ratio:.4f}")

            elif self.update_type == 'weighted':
                if self.selection_type == 'threshold':
                    if self.mode == 'gt':
                        weight = torch.clamp(param_data.abs() - self.threshold, min=0)
                    else:
                        weight = torch.clamp(self.threshold - param_data.abs(), min=0)
                else:
                    weight = mask.float() if mask is not None else torch.ones_like(param_data)

                total_w = self.total_weights[name]
                swa_param.data = (swa_data * total_w + param_data * weight) / (total_w + weight + 1e-8)
                self.total_weights[name] = total_w + weight

                if self.debug:
                    print(f"[DEBUG] {name} weighted stats: "
                          f"min={weight.min():.4e}, mean={weight.mean():.4e}, max={weight.max():.4e}")

        if self.update_type == 'masking':
            self.n += 1

    def apply_swa_weights(self, model):
        for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
            param.data.copy_(swa_param.data)

    def to(self, device):
        self.swa_model.to(device)

    def state_dict(self):
        return self.swa_model.state_dict()

    def load_state_dict(self, state_dict):
        self.swa_model.load_state_dict(state_dict)
