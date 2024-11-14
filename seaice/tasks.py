import asyncio
import datetime
import io
import random
import sys
from pathlib import Path

import numpy as np
from celery import shared_task
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sea_ice_backend import settings
from seaice.models import DownloadPredictTask
from seaice.osi_450_a import predict as predict_month
from seaice.osi_saf import predict
from seaice.osi_saf.data.download_and_organize_data import download_and_organize_data


def _download_predict_and_save(start_date_str, end_date_str, task_type):
    # 设置开始和结束日期
    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d")

    # 定义数据保存的目录
    output_directory = Path(settings.MEDIA_ROOT) / "downloads"

    # 调用下载并组织数据的函数
    nc_files = asyncio.run(download_and_organize_data(start_date, end_date, output_directory, task_type))
    print(nc_files)
    # 调用预测函数
    if task_type == 'DAILY':
        predictions = predict.predict_ice_concentration_from_nc_files(nc_files)
    else:
        # Initialize the current date to the start date
        current_date = start_date

        # List to store the months
        months = []

        # Loop through each month from start date to end date
        while current_date <= end_date:
            months.append(current_date.month)
            # Move to the next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        predictions = predict_month.predict_ice_concentration_from_nc_files(nc_files, months)

    # Save predictions as images and generate URLs
    result_urls = []
    for i, prediction in enumerate(predictions):
        # 生成带时间戳的随机文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_num = random.randint(1000, 9999)
        file_name = f"predict_task_{timestamp}_{random_num}.png"

        Land = ListedColormap(["#777777"])
        Land = np.repeat(np.array(Land(0))[None], 12, axis=0)
        blues = plt.get_cmap("Blues", 120)(np.linspace(0, 1, 120))[::-1]
        cmap = np.vstack((Land, blues))
        cmap = ListedColormap(cmap)

        # 转换预测结果为图片并应用colormap
        plt.imshow(prediction[0], cmap=cmap)
        plt.axis('off')  # 关闭坐标轴
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)

        # 保存文件
        file_path = Path("predicts") / file_name
        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                file_path, ContentFile(buffer.getvalue())
            )
        file_url = default_storage.url(file_path)
        result_urls.append(file_url)
        buffer.close()

    return [str(nc_file) for nc_file in nc_files], result_urls


@shared_task
def download_predict_and_save(task_type='DAILY'):
    if task_type == 'DAILY':
        # 获取当前日期的前一天作为结束日期
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        end_date_str = end_date.strftime("%Y%m%d")

        # 开始日期是结束日期的14天之前
        start_date = end_date - datetime.timedelta(days=14)
        start_date_str = start_date.strftime("%Y%m%d")
    else:
        # 获取当前日期的前一天作为结束日期
        end_date = datetime.datetime.now() - datetime.timedelta(days=1)
        end_date_str = end_date.strftime("%Y%m%d")

        # 开始日期是结束日期的12个月之前
        start_date = end_date - datetime.timedelta(days=365)
        start_date_str = start_date.strftime("%Y%m%d")

    # 创建数据库任务记录
    task = DownloadPredictTask.objects.create(
        start_date=start_date,
        end_date=end_date,
        task_type=task_type,
        source='SCHEDULED',
        status='IN_PROGRESS'
    )

    try:
        # 调用下载、预测和保存函数
        input_files, result_urls = _download_predict_and_save(start_date_str, end_date_str, task_type)
        task.input_files = input_files
        task.result_urls = result_urls
        task.status = 'COMPLETED'
    except Exception as e:
        task.status = 'FAILED'
        print(f"Error during task execution: {e}", file=sys.stderr)
    finally:
        task.save()

    return task.id


if __name__ == "__main__":
    # 获取当前日期的前一天作为结束日期
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date_str = end_date.strftime("%Y%m%d")

    # 开始日期是结束日期的12个月之前
    start_date = end_date - datetime.timedelta(days=14)
    start_date_str = start_date.strftime("%Y%m%d")
    input_files, result_urls = _download_predict_and_save(start_date_str, end_date_str, 'DAILY')
    print(input_files)
    print(result_urls)
