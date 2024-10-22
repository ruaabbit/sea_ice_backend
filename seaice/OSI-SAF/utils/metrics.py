import torch

Max_SIE = 25889.0


def mse_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    return mse.mean()


def rmse_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    rmse = torch.sqrt(mse)
    return rmse.mean()


def mae_func(pred, true, mask):
    mae = torch.abs(pred - true) * mask
    mae = torch.sum(mae, dim=[2, 3, 4]).mean(dim=1) / torch.sum(mask)
    return mae.mean()


def nse_func(pred, true, mask):
    squared_error = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1)
    mean_observation = torch.sum(true * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    mean_observation = (
        mean_observation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    squared_deviation = torch.sum(
        (true - mean_observation) ** 2 * mask, dim=[2, 3, 4]
    ).mean(dim=1)
    nse = 1 - squared_error / squared_deviation
    return nse.mean()


def PSNR_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    PSNR = 10 * torch.log10(1 * 1 / mse.mean())
    return PSNR


def BACC_func(pred, true, mask):
    # 使用布尔索引将大于0.15的位置设置为1，其他地方设置为0
    pred[pred > 0.15] = 1
    pred[pred <= 0.15] = 0
    true[true > 0.15] = 1
    true[true <= 0.15] = 0

    IIEE = torch.sum(torch.abs(pred - true) * mask, dim=[2, 3, 4]).mean(dim=1)
    BACC = 1 - IIEE.mean() / Max_SIE
    return BACC
