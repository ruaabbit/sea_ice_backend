import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import configs
from dataset.dataset import SIC_dataset
from train import Trainer
from utils.tools import setup_logging


def create_parser():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-st",
        "--start_time",
        type=int,
        required=True,
        help="Starting time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-et",
        "--end_time",
        type=int,
        required=True,
        help="Ending time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-save",
        "--save_result",
        action="store_true",
        help="Whether to save the test results",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    log_file = (
        f"{configs.test_results_path}/test_{configs.model}_{configs.input_length}.log"
    )
    logger = setup_logging(log_file)

    logger.info("\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("######################## Start testing! ########################")

    logger.info(f"Arguments:")
    logger.info(f"  start time: {args.start_time}")
    logger.info(f"  end time: {args.end_time}")
    logger.info(f"  output_dir: {configs.test_results_path}")
    logger.info(f"  data_paths: {configs.data_paths}")
    logger.info(f"  save_result: {args.save_result}")

    logger.info("Model Configurations:")
    logger.info(configs.__dict__)

    dataset_test = SIC_dataset(
        configs.data_paths,
        args.start_time,
        args.end_time,
        configs.input_gap,
        configs.input_length,
        configs.pred_shift,
        configs.pred_gap,
        configs.pred_length,
        samples_gap=1,
    )

    dataloader_test = DataLoader(dataset_test, shuffle=False)

    logger.info("Testing......")
    tester = Trainer()

    tester.network.load_state_dict(
        torch.load(
            f"checkpoints/checkpoint_{configs.model}_{configs.input_length}.pt",
            weights_only=True,
        )["net"]
    )

    mse, rmse, mae, nse, PSNR, BACC, loss = tester.vali(
        dataloader_test, torch.from_numpy(np.load("data/AMAP_mask.npy"))
    )
    logger.info(
        f"Metrics: mse: {mse:.5f}, rmse: {rmse:.5f}, mae: {mae:.5f}, nse: {nse:.5f}, PSNR: {PSNR:.5f}, BACC: {BACC:.5f}, loss: {loss:.5f}"
    )

    if args.save_result:
        sic_pred = tester.test(dataloader_test)

        logger.info(f"Saving output to {configs.test_results_path}")
        np.save(
            f"{configs.test_results_path}/sic_pred_{configs.model}.npy",
            sic_pred.cpu().numpy(),
        )
        np.save(f"{configs.test_results_path}/inputs.npy", dataset_test.get_inputs())
        np.save(f"{configs.test_results_path}/targets.npy", dataset_test.get_targets())
        np.save(f"{configs.test_results_path}/times.npy", dataset_test.get_times())

    logger.info("######################## End of test! ########################")
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
