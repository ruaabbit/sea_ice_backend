import logging
import numpy as np
from dateutil.relativedelta import relativedelta
import datetime
import pandas as pd
import xarray as xr


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def generate_date_list(start_time, end_time):
    """
    Generate a list of dates.

    Args:
        start_time: Start date in the format YYYYMMDD.
        end_time: End date in the format YYYYMMDD.

    Returns:
        List of dates in the format of list[YYYYMMDD].
    """
    start = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    return [
        int(dt.strftime("%Y%m%d"))
        for dt in [start + relativedelta(days=i) for i in range((end - start).days + 1)]
    ]


def time_features(dates, max_year, min_year):
    """
    Extract normalized time features (year, month, day) from a list of dates in YYYYMMDD format.

    Parameters
    ----------
        dates : List of dates in YYYYMMDD format.

    Returns
    -------
        Array of extracted time features (year, month, and day), normalized to [-0.5, 0.5].
    """
    dates = pd.to_datetime(dates, format="%Y%m%d")

    # 计算 year 特征，避免分母为零
    year_diff = max_year - min_year
    year_feature = (dates.year - min_year) / (year_diff if year_diff != 0 else 1)

    # 计算 month 和 day 特征
    month_feature = (dates.month - 1) / 11.0
    day_feature = (dates.day - 1) / 30.0

    # 按列堆叠特征
    return np.column_stack([year_feature, month_feature, day_feature])


def load_and_process_data(data_path):
    """
    Load and process data from a given path.

    Args:
        data_path: Path to the data file.

    Returns:
        Processed data.
    """
    data = xr.open_dataset(data_path)["z"].values
    return process_sea_ice_data(data)


def process_sea_ice_data(data):
    """
    0 - 100: Sea ice concentration %
    nan: Land and polar hole

    处理数据，包括归一化、处理缺失数据、陆地屏蔽等
    Args:
        data: 输入的海冰数据
    Returns:
        ice_conc: 处理后的海冰密集度数据
    """
    # 其他nan值进行置0
    ice_concentration = np.nan_to_num(data, nan=0)
    # 归一化
    ice_concentration /= 100.0
    assert not np.any(ice_concentration > 1) & np.any(np.isnan(ice_concentration))

    return ice_concentration


def prepare_input_target_indices(
    time_length,
    input_gap,
    input_length,
    prediction_shift,
    prediction_gap,
    prediction_length,
    sample_gap,
):
    """
    Args:
        input_gap: 两个连续输入帧之间的时间间隔
        input_length: 输入帧的数量
        prediction_shift: 最后一个目标预测的前导时间
        prediction_gap: 两个连续输出帧之间的时间间隔
        prediction_length: 输出帧的数量
        sample_gap: 两个检索样本的起始时间之间的间隔
    Returns:
        input_indices: 指向输入样本位置的索引
        target_indices: 指向目标样本位置的索引
    """
    assert prediction_shift >= prediction_length
    input_span = input_gap * (input_length - 1) + 1
    input_index = np.arange(0, input_span, input_gap)
    target_index = (
        np.arange(0, prediction_shift, prediction_gap) + input_span + prediction_gap - 1
    )
    indices = np.concatenate([input_index, target_index]).reshape(
        1, input_length + prediction_length
    )
    max_sample_count = time_length - (input_span + prediction_shift - 1)
    indices = indices + np.arange(max_sample_count)[:, np.newaxis] @ np.ones(
        (1, input_length + prediction_length), dtype=int
    )
    input_indices = indices[::sample_gap, :input_length]
    target_indices = indices[::sample_gap, input_length:]
    assert len(input_indices) == len(target_indices)
    return input_indices, target_indices
