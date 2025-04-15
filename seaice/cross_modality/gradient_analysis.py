import argparse
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Configs
from dataset.dataset import SIC_dataset
from train import MyLightningModule
from utils.metrics import *


def calculate_daily_gradients(
    model,
    dataloader,
    device,
    pred_gap,
    grad_type="sum",
    position=None,
    variable=None,
):
    """逐日预测梯度计算，支持区域分析和特定像素位置分析"""
    model.eval()
    pptvs = []

    # 创建像素位置掩码
    if position is not None and len(position) >= 2:
        pixel_mask = torch.zeros((1, 1, 1, 448, 304), device=device)
        try:
            # 计算矩形区域的边界
            x_coords = [x for x, y in position]
            y_coords = [y for x, y in position]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            # 将矩形区域内的所有像素点设置为1
            pixel_mask[..., x_min : x_max + 1, y_min : y_max + 1] = 1.0
        except Exception as e:
            print(f"创建像素掩码时出错: {e}")
            # 如果出错，使用全1掩码（不进行区域限制）
            pixel_mask = torch.ones((1, 1, 1, 448, 304), device=device)

    for inputs, targets, inputs_mark, targets_mark in dataloader:
        # 数据准备
        inputs = inputs.to(device).requires_grad_(True)
        targets = targets.to(device)
        mask = model.mask.to(device)

        # 前向传播
        pred = model(inputs, inputs_mark, targets_mark)
        # 不打印每个批次的形状，减少输出冗余
        # print(pred.shape)

        # 梯度计算
        if position is not None:
            # 只关注矩形区域内的预测
            pred = pred * pixel_mask

        # 选择特定的预测时间步长
        if pred_gap is not None:
            pred = pred[:, pred_gap - 1, :, :, :]

        # 选择特定的变量
        if variable is not None:
            pred = pred[:, variable - 1, :, :]

        # 计算梯度
        if grad_type == "sum":
            f = torch.sum(torch.abs(pred) * mask*25)
        else:  # l2
            f = torch.sum((pred * mask ) ** 2)

        # 反向传播
        f.backward()

        # 收集梯度绝对值
        grad = inputs.grad
        pptv = grad.abs().cpu().numpy()
        pptvs.append(pptv)

        # 清除梯度
        inputs.grad = None

    return np.concatenate(pptvs, axis=0)


# 全局归一化
def global_normalization(data):
    data_min = np.min(data)
    data_max = np.max(data)
    normalized = (data - data_min) / (data_max - data_min + 1e-8)
    return normalized


# 逐通道归一化
def channelwise_normalization(data):
    # 对每个通道单独归一化
    normalized = np.zeros_like(data)
    for c in range(data.shape[0]):
        channel_data = data[c]
        data_min = np.min(channel_data)
        data_max = np.max(channel_data)
        normalized[c] = (channel_data - data_min) / (data_max - data_min + 1e-8)
    return normalized


def plot_channel_gradients(grad_data, channel_names=None, save_dir="./", filename="average_gradients"):
    """
    可视化梯度数据
    参数：
        grad_data     : numpy数组，形状为 (通道数, 高, 宽)
        channel_names : 各通道名称列表
        save_dir      : 保存路径
        filename      : 保存的文件名
    """
    os.makedirs(save_dir, exist_ok=True)

    if channel_names is None:
        num_channels = grad_data.shape[0]
        channel_names = [f"Channel_{i + 1}" for i in range(num_channels)]

    # 创建单张大图
    plt.figure(figsize=(6 * grad_data.shape[0], 6))

    for ch_idx in range(grad_data.shape[0]):
        ax = plt.subplot(1, grad_data.shape[0], ch_idx + 1)
        data = grad_data[ch_idx]  # 直接取通道数据

        # 绘制热力图
        im = ax.imshow(data, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 统计信息
        stats_text = f"""Mean: {data.mean():.2e}
        Max: {data.max():.2e}
        Min: {data.min():.2e}"""
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))

        ax.set_title(f"{channel_names[ch_idx]}")
        ax.axis('off')

    plt.savefig(f"{save_dir}/{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()


def process_gradients():
    """处理梯度文件夹中的.npy文件，对它们进行求和、平均并可视化"""
    try:
        # 设置路径
        # 使用绝对路径确保能找到文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        gradients_dir = os.path.join(current_dir, "gradient_analysis/gradients")
        output_dir = os.path.join(current_dir, "gradient_analysis/results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有梯度文件
        gradient_files = glob.glob(os.path.join(gradients_dir, "gradients_day*.npy"))
        print(f"找到 {len(gradient_files)} 个梯度文件")
        
        if not gradient_files:
            print("未找到梯度文件，请检查路径")
            return None
        
        try:
            # 加载第一个文件以获取形状信息
            sample_data = np.load(gradient_files[0])
            print(f"梯度文件形状: {sample_data.shape}")
            
            # 初始化累加数组
            # 预期形状为 [1, 7, 6, 448, 304]
            # 我们需要对所有文件求和，所以初始化一个相同形状的零数组
            accumulated_gradients = np.zeros_like(sample_data)
        except Exception as e:
            print(f"加载样本梯度文件失败: {e}")
            return None
        
        # 累加所有梯度文件
        print("正在累加梯度文件...")
        processed_files = 0
        for file_path in tqdm(gradient_files):
            try:
                gradient_data = np.load(file_path)
                accumulated_gradients += gradient_data
                processed_files += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}，跳过此文件")
                continue
        
        if processed_files == 0:
            print("没有成功处理任何梯度文件")
            return None
        
        # 计算平均梯度
        average_gradients = accumulated_gradients / processed_files
        
        # 对平均梯度进行归一化
        average_gradients = channelwise_normalization(average_gradients)
        print(f"平均梯度形状: {average_gradients.shape}")
        
        # 保存平均梯度
        try:
            avg_save_path = os.path.join(output_dir, "average_gradients.npy")
            np.save(avg_save_path, average_gradients)
            print(f"平均梯度已保存至: {avg_save_path}")
        except Exception as e:
            print(f"保存平均梯度失败: {e}")
        
        # 可视化平均梯度
        try:
            # 假设形状为 [1, 7, 6, 448, 304]，我们需要对每个时间步的每个通道进行可视化
            # 首先对整个序列求平均，得到 [1, 6, 448, 304]
            sequence_avg = np.mean(average_gradients, axis=1)
            
            # 去掉批次维度，得到 [6, 448, 304]
            sequence_avg = sequence_avg.squeeze(0)
            
            # 定义通道名称
            channel_names = ["sic", "si_u", "si_v", "t2m", "u10", "v10"]
            
            # 可视化平均梯度
            print("正在可视化平均梯度...")
            plot_channel_gradients(
                grad_data=sequence_avg,
                channel_names=channel_names,
                save_dir=output_dir,
                filename="average_gradients_all_timesteps"
            )
            
            # 对每个时间步分别可视化
            for t in range(average_gradients.shape[1]):
                # 提取当前时间步的数据 [1, 6, 448, 304]
                timestep_data = average_gradients[:, t, :, :, :]
                # 去掉批次维度 [6, 448, 304]
                timestep_data = timestep_data.squeeze(0)
                
                # 可视化当前时间步
                plot_channel_gradients(
                    grad_data=timestep_data,
                    channel_names=channel_names,
                    save_dir=output_dir,
                    filename=f"average_gradients_timestep_{t}"
                )
            
            print("可视化完成！")
        except Exception as e:
            print(f"可视化平均梯度失败: {e}")
        
        print("处理完成！")
        return average_gradients
    except Exception as e:
        print(f"处理梯度文件过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def grad_nb(
    model,
    device,
    start_time: int,
    end_time: int,
    pred_gap: int,
    grad_type: str,
    position: str,
    variable: int = None,
):
    """计算指定时间范围内的梯度，可选择保存或直接可视化结果"""
    try:
        variables = ["SIC", "SI_U", "SI_V", "T2M", "U10M", "V10M"]
        config = Configs()
        torch.set_float32_matmul_precision("high")        
        print(f"\n加载模型和数据...")
        with torch.cuda.device(device):
            # 创建专用数据加载器
            try:
                grad_dataset = SIC_dataset(
                    config.full_data_path,
                    start_time,
                    end_time,
                    config.input_gap,
                    config.input_length,
                    config.pred_shift,
                    config.pred_gap,
                    config.pred_length,
                    samples_gap=1,
                )
                grad_loader = DataLoader(grad_dataset, batch_size=1, shuffle=False)
                print("数据加载完成!")
            except Exception as e:
                print(f"数据加载失败: {e}")
                raise
            
            # 解析像素位置
            pixel_positions = None
            if position:
                pixel_positions = [tuple(map(int, pos.split(","))) 
                                for pos in position.split(";")]
                print(f"分析范围: {pixel_positions}")

            
            # 打印分析参数
            param_info = []
            if pred_gap:
                param_info.append(f"预测时间步: {pred_gap}")
            if variable:
                param_info.append(f"分析变量: {variables[variable-1]}")
            
            # 执行梯度计算
            gradients = calculate_daily_gradients(
                model,
                grad_loader,
                device,
                pred_gap=pred_gap,
                grad_type=grad_type,
                position=pixel_positions,
                variable=variable
            )
            print(f"梯度计算完成，形状: {gradients.shape}")
            
            return gradients
    except Exception as e:
        print(f"梯度计算过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None



def batch_process_gradients(model,device,start_date, end_date, pred_gap, grad_type="sum", position=None, variable=None):
    """
    批量处理从起始日期到结束日期的每一天的梯度计算
    
    参数：
        start_date  : 起始日期，格式为YYYYMMDD
        end_date    : 结束日期，格式为YYYYMMDD
        pred_gap    : 预测提前期，范围是1-7
        grad_type   : 梯度计算方式，"sum"或"l2"
        position    : 要分析的像素位置，格式为'x1,y1;x2,y2;x3,y3;x4,y4'
        variable    : 要分析的变量，分别为1-6[SIC,SI_U,SI_V,T2M,U10M,V10M]
        save_result : 是否保存梯度结果
        visualize   : 是否可视化梯度结果
    """
    import datetime
    
    # 配置参数
    input_length = 7
    prediction_length = 7
    days_to_add = input_length + prediction_length - 1
    
    # 将日期字符串转换为datetime对象
    start_date_obj = datetime.datetime.strptime(str(start_date), "%Y%m%d")
    end_date_obj = datetime.datetime.strptime(str(end_date), "%Y%m%d")
    
    # 计算调整后的结束日期（减去days_to_add）
    adjusted_end_date_obj = end_date_obj - datetime.timedelta(days=days_to_add)
    
    # 初始化累积梯度数组（如果需要累积结果）
    accumulated_gradients = None
    gradient_count = 0
    
    # 在日期范围内循环
    current_date_obj = start_date_obj
    while current_date_obj <= adjusted_end_date_obj:
        # 获取当前日期的字符串表示
        current_start_date = current_date_obj.strftime("%Y%m%d")
        current_end_date_obj = current_date_obj + datetime.timedelta(days=days_to_add)
        current_end_date = current_end_date_obj.strftime("%Y%m%d")
        
        print(f"\n处理日期范围: {current_start_date} 到 {current_end_date}")
        
        # 执行梯度计算
        try:
            gradients = grad_nb(
                model=model,
                device=device,
                start_time=int(current_start_date),
                end_time=int(current_end_date),
                pred_gap=pred_gap,
                grad_type=grad_type,
                position=position,
                variable=variable
            )
            
            # 如果需要累积结果
            if accumulated_gradients is None:
                accumulated_gradients = gradients.copy()
            else:
                accumulated_gradients += gradients
            
            gradient_count += 1
            
        except Exception as e:
            print(f"处理日期 {current_start_date} 时出错: {e}")
            # 继续处理下一天，而不是中断整个过程
        
        # 移动到下一天
        current_date_obj += datetime.timedelta(days=1)
    
    # 如果有累积的梯度数据，计算平均值并可视化
    if accumulated_gradients is not None and gradient_count > 0:
        # 计算平均梯度
        average_gradients = accumulated_gradients / gradient_count
        
        # 创建输出目录
        output_dir = os.path.join("gradient_analysis/results", f"average_{start_date}_to_{end_date}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义通道名称
        channel_names = ["sic", "si_u", "si_v", "t2m", "u10", "v10"]
        
        # 对梯度数据进行归一化
        normalized_gradients = channelwise_normalization(average_gradients.squeeze(0))
        
        # 可视化梯度数据
            
        # 对每个时间步分别可视化
        for t in range(normalized_gradients.shape[0]):
            # 提取当前时间步的数据
            timestep_data = normalized_gradients[t]
            
            # 可视化当前时间步
            plot_channel_gradients(
                grad_data=timestep_data,
                channel_names=channel_names,
                save_dir=output_dir,
                filename=f"average_gradients_{start_date}_to_{end_date}_timestep_{t}"
            )
        
        print(f"平均梯度可视化结果已保存至：{output_dir}")
    print("批处理完成！")
    return accumulated_gradients, gradient_count


def main(start_time, end_time, pred_gap=1, grad_type="sum", position=None, variable=None):
    """
    梯度分析主函数，可以作为模块被其他脚本调用
    
    参数：
        start_time  : 起始时间 (八位数字, YYYYMMDD)
        end_time    : 结束时间 (八位数字, YYYYMMDD)
        pred_gap    : 预测提前期，范围是1-7
        grad_type   : 梯度计算方式，"sum"或"l2"
        position    : 要分析的像素范围，格式为'x1,y1;x2,y2;x3,y3;x4,y4'
        variable    : 要分析的变量，分别为1-6[SIC,SI_U,SI_V,T2M,U10M,V10M]
    
    返回：
        tuple: (accumulated_gradients, gradient_count) 累积梯度和处理的梯度文件数量
    """
    try:
        Model_Type = "SimVP_7_7"
        checkpoint_path = f"checkpoints/{Model_Type}.ckpt"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyLightningModule.load_from_checkpoint(
                    checkpoint_path, map_location=device
                ).to(device)
        mask = model.mask.to(device)
        print("模型加载完成!")
        print(f"开始梯度分析: {start_time} 到 {end_time}")
        print(f"参数: pred_gap={pred_gap}, grad_type={grad_type}, variable={variable if variable else '全部'}")
        
        # 使用批处理模式，处理从start_time到end_time的所有日期
        accumulated_gradients, gradient_count = batch_process_gradients(
            model=model,
            device=device,
            start_date=start_time,
            end_date=end_time,
            pred_gap=pred_gap,
            grad_type=grad_type,
            position=position,
            variable=variable
        )
        
        print(f"梯度分析完成: 共处理 {gradient_count} 个日期")
        return accumulated_gradients, gradient_count
    
    except Exception as e:
        print(f"梯度分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


if __name__ == "__main__":
    start_time = 20220101
    end_time = 20231231
    pred_gap = None
    grad_type = "sum"
    position = None
    variable = None
    main(
        start_time,
        end_time,
        grad_type="sum",
        pred_gap=pred_gap,
        position=position,
        variable=variable
    )