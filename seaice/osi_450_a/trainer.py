import numpy as np
import torch

from .model_factory import IceNet


class Trainer:

    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.arctic_mask = torch.from_numpy(np.load("seaice/osi_450_a/data/ocean_mask.npy"))

        self._build_network()

    def _build_network(self):
        self.network = IceNet(self.configs).to(self.device)

    def get_grad(self, dataloader, grad_month, grad_type):
        sic_pred_list = []
        self.network.eval()
        a = grad_month
        b = grad_month + 1
        for inputs, targets, input_times in dataloader:
            inputs = inputs.float().to(self.device)
            inputs.requires_grad = True
            targets = targets.float().to(self.device)
            arctic_mask = self.arctic_mask.to(self.device)

            input_times = torch.tensor(input_times, dtype=torch.float32).to(self.device)

            outputgrad, _ = self.network(inputs, targets, input_times)
            outputgrad = outputgrad[0, a:b, 0, :, :]
            outputgrad = outputgrad[0, :, :]

            input = inputs[0, a:b, 0, :, :]
            input = input[0, :, :]

            if grad_type == "sum":
                outputgrad_1 = torch.sum(abs(input - outputgrad) * arctic_mask)
            else:
                outputgrad_1 = torch.sqrt(
                    torch.sum(((input - outputgrad) * arctic_mask) ** 2)
                )

            outputgrad_1.backward()
            grads = inputs.grad.cpu().numpy()
            gradsabs = abs(grads)

            sic_pred_list.append(gradsabs)

        return sic_pred_list
