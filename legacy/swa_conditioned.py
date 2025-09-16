import torch
from copy import deepcopy

class StepwiseConditionalSWA:
    def __init__(self, model, threshold=0.7, mode='gt', n_steps=None, n_epochs=None):
        assert (n_steps is not None) ^ (n_epochs is not None), \
            "Either n_steps or n_epochs must be set, not both"

        self.swa_model = deepcopy(model)
        self.threshold = threshold
        self.mode = mode
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.step_counter = 0
        self.epoch_counter = 0
        self.n = 0  # update回数

    def maybe_update(self, model, current_step=None, current_epoch=None):
        # タイミング確認
        if self.n_steps is not None:
            self.step_counter += 1
            if self.step_counter % self.n_steps != 0:
                return
        elif self.n_epochs is not None:
            if current_epoch is None or (current_epoch % self.n_epochs != 0):
                return

        # パラメータ平均（しきい値条件付き）
        for (name, swa_param), (_, model_param) in zip(
            self.swa_model.named_parameters(), model.named_parameters()
        ):
            new_data = model_param.data
            current = swa_param.data

            if self.mode == 'gt':
                # mask = new_data > self.threshold
                mask = torch.ones_like(new_data, dtype=torch.bool)  # 全体平均にする
            elif self.mode == 'lt':
                # mask = new_data < self.threshold
                mask = torch.ones_like(new_data, dtype=torch.bool)  # 全体平均にする
            else:
                raise ValueError("mode must be 'gt' or 'lt'")

            if mask.any():
                current[mask] = (current[mask] * self.n + new_data[mask]) / (self.n + 1)

        self.n += 1

    # def maybe_update(self, model, current_step=None, current_epoch=None):
    #     updated = False

    #     for (name, swa_param), (_, model_param) in zip(
    #         self.swa_model.named_parameters(), model.named_parameters()
    #     ):
    #         new_data = model_param.data
    #         current = swa_param.data

    #         if self.mode == 'gt':
    #             # mask = new_data > self.threshold
    #             mask = torch.ones_like(new_data, dtype=torch.bool)  # 全体平均にする
    #         elif self.mode == 'lt':
    #             # mask = new_data < self.threshold
    #             mask = torch.ones_like(new_data, dtype=torch.bool)  # 全体平均にする
    #         else:
    #             raise ValueError("mode must be 'gt' or 'lt'")

    #         if mask.any():
    #             current[mask] = (current[mask] * self.n + new_data[mask]) / (self.n + 1)
    #             updated = True

    #     if updated:
    #         self.n += 1


    def apply_swa_weights(self, model):
        for swa_param, model_param in zip(self.swa_model.parameters(), model.parameters()):
            model_param.data.copy_(swa_param.data)

    def state_dict(self):
        return self.swa_model.state_dict()

    def load_state_dict(self, state_dict):
        self.swa_model.load_state_dict(state_dict)


