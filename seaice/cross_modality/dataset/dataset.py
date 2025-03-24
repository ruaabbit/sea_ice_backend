import numpy as np
from torch.utils.data import Dataset
import datetime
from ..utils import (
    generate_date_list,
    time_features,
    prepare_input_target_indices,
)


class SIC_dataset(Dataset):
    def __init__(
        self,
        file_with_paths,
        start_time,
        end_time,
        input_gap,
        input_length,
        pred_shift,
        pred_gap,
        pred_length,
        samples_gap=1,
    ):
        super().__init__()
        self.time_list = generate_date_list(start_time, end_time)

        # 计算索引列表
        self.input_indices, self.target_indices = prepare_input_target_indices(
            len(self.time_list),
            input_gap,
            input_length,
            pred_shift,
            pred_gap,
            pred_length,
            samples_gap,
        )

        # 加载变量的路径
        self.variables = {
            "sic": np.genfromtxt(file_with_paths["sic"], dtype=str),
            "siv_u": np.genfromtxt(file_with_paths["siv_u"], dtype=str),
            "siv_v": np.genfromtxt(file_with_paths["siv_v"], dtype=str),
            "u10": np.genfromtxt(file_with_paths["u10"], dtype=str),
            "v10": np.genfromtxt(file_with_paths["v10"], dtype=str),
            "t2m": np.genfromtxt(file_with_paths["t2m"], dtype=str),
        }

        self.max_values = np.load(file_with_paths["max"])
        self.min_values = np.load(file_with_paths["min"])

        # 获取sic整个数据集上的起始日期和结束日期
        self.time_coords_begin = self.variables["sic"][0].split("_")[-1].split(".")[0]
        self.time_coords_end = self.variables["sic"][-1].split("_")[-1].split(".")[0]

        self.data_stamp = time_features(self.time_list)

        self.offset = (
            datetime.datetime.strptime(str(start_time), "%Y%m%d")
            - datetime.datetime.strptime(str(self.time_coords_begin), "%Y%m%d")
        ).days

    def _normalize(self, data, variable_index):
        """
        将数据归一化到 [0, 1] 之间。
        """
        min_val = self.min_values[variable_index]
        max_val = self.max_values[variable_index]

        if max_val == min_val:
            raise ValueError("max_val 和 min_val 不能相同")

        return (data - min_val) / (max_val - min_val)

    def _denormalize(self, normalized_data, variable_index):
        """
        将归一化后的数据反归一化回原始范围。
        """
        min_val = self.min_values[variable_index]
        max_val = self.max_values[variable_index]

        if max_val == min_val:
            raise ValueError("max_val 和 min_val 不能相同")

        return normalized_data * (max_val - min_val) + min_val

    def _load_data(self, paths, variable_index, start_idx, end_idx):
        """加载并归一化数据"""
        return np.array(
            [
                self._normalize(np.load(path), variable_index)
                for path in paths[start_idx : end_idx + 1]
            ]
        )[
            :, None
        ]  # 形状: (T, 1, H, W)

    def _get_data(self, indices):
        """获取输入或目标数据"""
        start_idx = indices[0] + self.offset
        end_idx = indices[-1] + self.offset

        data = []
        for i, key in enumerate(self.variables):
            data.append(self._load_data(self.variables[key], i, start_idx, end_idx))

        return np.concatenate(data, axis=1)  # 形状: (T, 6, H, W)

    def __len__(self):
        return len(self.input_indices)

    def __getitem__(self, index):
        # 获取输入和目标数据
        inputs = self._get_data(self.input_indices[index])
        targets = self._get_data(self.target_indices[index])

        # 获取时间特征
        inputs_mark = self.data_stamp[
            self.input_indices[index][0] : self.input_indices[index][-1] + 1
        ]
        targets_mark = self.data_stamp[
            self.target_indices[index][0] : self.target_indices[index][-1] + 1
        ]

        return inputs, targets, inputs_mark, targets_mark

    def get_inputs(self):
        return self._get_data(self.input_indices[0])

    def get_targets(self):
        return self._get_data(self.target_indices[0])

    def get_times(self):
        return self.time_list
