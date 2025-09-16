import torch
from copy import deepcopy

class StepwiseConditionalSWA:
    def __init__(
        self, model, threshold=0.0, mode='gt',
        update_type='masking',
        selection_type='threshold', topk_ratio=0.1,
        n_steps=None, n_epochs=None
    ):
        assert update_type in ['masking', 'weighted']
        assert selection_type in ['threshold', 'topk', 'bottomk', 'none']
        if selection_type == 'threshold':
            assert mode in ['gt', 'lt']
        assert (n_steps is not None) ^ (n_epochs is not None), "Specify either n_steps or n_epochs"

        self.swa_model = deepcopy(model)
        self.threshold = threshold  # could be float or str like 'mean+1std'
        self.mode = mode
        self.update_type = update_type
        self.selection_type = selection_type
        self.topk_ratio = topk_ratio
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.step_counter = 0
        self.n = 0  # for masking average count
        self.debug = True

        # for weighted mode
        self.total_weights = {
            name: torch.zeros_like(p.data)
            for name, p in self.swa_model.named_parameters()
        }

    def _resolve_dynamic_threshold(self, param):
        flat = param.abs().view(-1)
        if isinstance(self.threshold, str):
            if self.threshold == 'mean':
                return flat.mean()
            elif self.threshold.startswith('mean+'):
                coeff = float(self.threshold.split('+')[1].replace('std', ''))
                return flat.mean() + coeff * flat.std()
            elif self.threshold.startswith('mean-'):
                coeff = float(self.threshold.split('-')[1].replace('std', ''))
                return flat.mean() - coeff * flat.std()
            elif self.threshold == 'median':
                return flat.median()
            elif self.threshold.startswith('percentile'):
                p = float(self.threshold.replace('percentile', ''))
                return torch.quantile(flat, p / 100.0)
            else:
                try:
                    return float(self.threshold)
                except ValueError:
                    raise ValueError(f"Unsupported threshold format: {self.threshold}")
        else:
            return self.threshold  # numeric case

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

            # --- マスク作成 ---
            if self.selection_type == 'threshold':
                threshold_value = self._resolve_dynamic_threshold(param_data)
                if self.mode == 'gt':
                    mask = param_data.abs() > threshold_value
                else:
                    mask = param_data.abs() < threshold_value

            elif self.selection_type in ['topk', 'bottomk']:
                abs_param = param_data.abs().flatten()
                k = int(self.topk_ratio * abs_param.numel())
                if k == 0:
                    continue
                topk_vals, topk_indices = torch.topk(
                    abs_param,
                    k=k,
                    largest=(self.selection_type == 'topk')
                )
                mask = torch.zeros_like(abs_param, dtype=torch.bool)
                mask[topk_indices] = True
                mask = mask.view_as(param_data)
            else:
                raise ValueError(f"Unknown selection_type: {self.selection_type}")
            
            if self.debug:
                mask_ratio = mask.sum().item() / mask.numel()
                print(f"[DEBUG] {name} mask ratio: {mask_ratio:.4f} | shape: {param_data.shape}")
                

            # --- 平均更新 ---
            if self.update_type == 'masking':
                swa_data[mask] = (swa_data[mask] * self.n + param_data[mask]) / (self.n + 1)

            elif self.update_type == 'weighted':
                if self.selection_type == 'threshold':
                    abs_param = param_data.abs()
                    threshold_value = self._resolve_dynamic_threshold(param_data)
                    max_val = abs_param.max() + 1e-8

                    if self.mode == 'gt':
                        raw_weight = abs_param - threshold_value
                    else:
                        raw_weight = threshold_value - abs_param

                    weight = torch.clamp(raw_weight / (max_val - threshold_value + 1e-8), min=0.0)
                elif self.selection_type in ['topk', 'bottomk'] and self.update_type == 'weighted':
                    flat_abs = param_data.abs().flatten()
                    percentile = self.topk_ratio
                    k = int(percentile * flat_abs.numel())

                    if k == 0:
                        if self.debug:
                            print(f"[DEBUG] {name} skipped due to k=0")
                        continue

                    threshold_value = torch.kthvalue(flat_abs, flat_abs.numel() - k).values if self.selection_type == 'topk' else torch.kthvalue(flat_abs, k).values
                    max_val = flat_abs.max() + 1e-6
                    norm_weight = (param_data.abs() - threshold_value) / (max_val - threshold_value + 1e-6)
                    weight = torch.clamp(norm_weight, min=0.0)

                total_w = self.total_weights[name]
                swa_param.data = (swa_data * total_w + param_data * weight) / (total_w + weight + 1e-8)
                self.total_weights[name] = total_w + weight

                if self.debug:
                    print(f"[DEBUG] {name} weight stats -> min: {weight.min():.4e}, mean: {weight.mean():.4e}, max: {weight.max():.4e}")

        # if self.update_type == 'masking':
        #     self.n += 1
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
