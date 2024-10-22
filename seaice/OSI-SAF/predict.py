import torch
from torch.amp import autocast
import numpy as np
from utils.model_factory import IceNet
from config import configs


class Predictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0")
        self.network = IceNet().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.network.load_state_dict(checkpoint["net"])
        self.network.eval()

        self.arctic_mask = torch.from_numpy(np.load("data/AMAP_mask.npy")).to(self.device)

    def predict(self, inputs):
        inputs = torch.from_numpy(inputs).float().to(self.device)

        with torch.no_grad():
            with autocast(device_type="cuda"):
                sic_pred, _ = self.network(inputs, None)  # None表示预测时不需要目标值

        return sic_pred.cpu().numpy()


if __name__ == "__main__":
    # 加载一些示例输入数据
    example_inputs = np.load("path/to/example_inputs.npy")

    # 模型路径
    model_path = "checkpoints/checkpoint_SICFN_14.pt"

    # 创建Predictor实例
    predictor = Predictor(model_path)

    # 生成预测
    predictions = predictor.predict(example_inputs)

    # 保存或返回预测结果
    np.save("path/to/save_predictions.npy", predictions)
