import asyncio
import datetime
import json
from pathlib import Path

from PIL import Image
from celery import shared_task
from dateutil.relativedelta import relativedelta

from sea_ice_backend import settings
from seaice.common.convert_data_and_generate_image import (
    prediction_result_to_image,
    prediction_result_to_globe_image,
)
from seaice.common.download_and_organize_data import download_and_organize_data
from seaice.cross_modality.model_interpreter import grad_nb as grad_nb_day
from seaice.models import (
    DownloadPredictTask,
    DynamicGradTask,
    ModelInterpreterTask,
    DownloadPredictGlobeTask,
)
from seaice.osi_450_a import predict as predict_month
from seaice.osi_450_a.grad import grad_nb as grad_nb_month
from seaice.osi_saf import predict


def _download_and_save(start_date_str, end_date_str, task_type):
    # 设置开始和结束日期
    start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date_str, "%Y%m%d")
    # 定义数据保存的目录
    output_directory = Path(settings.MEDIA_ROOT) / "downloads"
    # 调用下载并组织数据的函数
    nc_files = asyncio.run(
        download_and_organize_data(start_date, end_date, output_directory, task_type)
    )
    input_times = []
    if task_type == "DAILY":
        # 生成输入时间列表
        current_date = start_date
        while current_date <= end_date:
            input_times.append(int(current_date.strftime("%Y%m%d")))
            current_date = current_date + datetime.timedelta(days=1)

    elif task_type == "MONTHLY":
        # 生成输入时间列表
        current_date = start_date
        while current_date <= end_date:
            input_times.append(int(current_date.strftime("%Y%m")))
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
    else:
        raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")

    return [str(nc_file) for nc_file in nc_files], input_times


def _predict_and_save(input_files, input_times, task_type):
    # 调用预测函数
    if task_type == "DAILY":
        predictions = predict.predict_ice_concentration_from_nc_files(input_files)
    elif task_type == "MONTHLY":
        predictions = predict_month.predict_ice_concentration_from_nc_files(
            input_files, [int(time % 100) for time in input_times]
        )
    else:
        raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")
    # Save predictions as images and generate URLs
    result_urls = []
    for prediction in predictions:
        file_url = prediction_result_to_image(prediction[0])
        result_urls.append(file_url)

    return result_urls


def _predict_and_save_globe(input_files, input_times, task_type):
    # 调用预测函数
    if task_type == "DAILY":
        predictions = predict.predict_ice_concentration_from_nc_files(input_files)
    elif task_type == "MONTHLY":
        predictions = predict_month.predict_ice_concentration_from_nc_files(
            input_files, [int(time % 100) for time in input_times]
        )
    else:
        raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")
    # Save predictions as images and generate URLs
    result_urls = []
    for prediction in predictions:
        file_url = prediction_result_to_globe_image(prediction[0])
        result_urls.append(file_url)

    return result_urls


@shared_task
def predict_and_return(input_images_paths, input_times, task_type, task_id):
    task = DownloadPredictTask.objects.get(id=task_id)
    # Open images from local paths
    images = []
    for path_str in input_images_paths:
        try:
            with Image.open(path_str) as img:
                images.append(img.copy())
        except Exception as e:
            task.status = "FAILED"
            task.save()
            raise ValueError(f"Failed to open image {path_str}: {str(e)}")
    task.input_files = input_images_paths
    task.input_times = input_times

    if task_type == "DAILY":
        predictions = predict.predict_ice_concentration_from_images(images)
    elif task_type == "MONTHLY":
        predictions = predict_month.predict_ice_concentration_from_images(
            images, input_times
        )
    else:
        task.status = "FAILED"
        task.save()
        raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")
    urls = []
    for prediction in predictions:
        file_url = prediction_result_to_image(prediction[0])
        urls.append(file_url)
    task.result_urls = urls
    task.status = "COMPLETED"
    task.save()
    return json.dumps(
        {
            "input_files": task.input_files,
            "input_times": task.input_times,
            "result_urls": task.result_urls,
            "task_id": task.id,
            "status": task.status,
        }
    )


@shared_task
def predict_and_return_globe(input_images_paths, input_times, task_type, task_id):
    task = DownloadPredictGlobeTask.objects.get(id=task_id)
    # Open images from local paths
    images = []
    error = ""
    for path_str in input_images_paths:
        try:
            with Image.open(path_str) as img:
                images.append(img.copy())
        except Exception as e:
            task.status = "FAILED"
            task.save()
            raise ValueError(f"Failed to open image {path_str}: {str(e)}")
    task.input_files = input_images_paths
    task.input_times = input_times
    try:
        if task_type == "DAILY":
            predictions = predict.predict_ice_concentration_from_images(images)
        elif task_type == "MONTHLY":
            predictions = predict_month.predict_ice_concentration_from_images(
                images, input_times
            )
        else:
            task.status = "FAILED"
            task.save()
            raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")
        urls = []
        for prediction in predictions:
            file_url = prediction_result_to_globe_image(prediction[0])
            urls.append(file_url)
        task.result_urls = urls
        task.status = "COMPLETED"
    except Exception as e:
        task.status = f"FAILED"
        error = e
    finally:
        task.save()
    return json.dumps(
        {
            "input_files": task.input_files,
            "input_times": task.input_times,
            "result_urls": task.result_urls,
            "task_id": task.id,
            "status": task.status,
            "error": str(error),
        }
    )


@shared_task
def grad_and_return(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    grad_month: int,
    grad_type: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    task_id: int,
):
    task = DynamicGradTask.objects.get(id=task_id)
    error = ""
    task.start_date = start_time
    task.end_date = end_time
    task.grad_month = grad_month
    task.grad_type = grad_type
    task.x1 = x1
    task.y1 = y1
    task.x2 = x2
    task.y2 = y2

    start_time = int(start_time.strftime("%Y%m"))
    end_time = int(end_time.strftime("%Y%m"))

    try:
        result_urls = grad_nb_month(
            start_time, end_time, grad_month, grad_type, x1, y1, x2, y2
        )
        task.result_urls = result_urls
        task.status = "COMPLETED"
    except Exception as e:
        task.status = "FAILED"
        error = e
    finally:
        task.save()

    return json.dumps(
        {
            "start_time": task.start_date.strftime("%Y%m"),
            "end_time": task.end_date.strftime("%Y%m"),
            "grad_month": task.grad_month,
            "grad_type": task.grad_type,
            "x1": task.x1,
            "y1": task.y1,
            "x2": task.x2,
            "y2": task.y2,
            "result_urls": task.result_urls,
            "task_id": task.id,
            "status": task.status,
            "error": str(error),
        }
    )


@shared_task
def grad_day_and_return(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    grad_day: int,
    grad_type: str,
    task_id: int,
):
    task = ModelInterpreterTask.objects.get(id=task_id)
    error = ""
    task.start_date = start_time
    task.end_date = end_time
    task.grad_day = grad_day
    task.grad_type = grad_type

    start_time = int(start_time.strftime("%Y%m%d"))
    end_time = int(end_time.strftime("%Y%m%d"))

    try:
        result_urls = grad_nb_day(start_time, end_time, grad_day, grad_type)
        task.result_urls = result_urls
        task.status = "COMPLETED"
    except Exception as e:
        task.status = "FAILED"
        error = e
    finally:
        task.save()

    return json.dumps(
        {
            "start_time": task.start_date.strftime("%Y%m%d"),
            "end_time": task.end_date.strftime("%Y%m%d"),
            "grad_day": task.grad_day,
            "grad_type": task.grad_type,
            "result_urls": task.result_urls,
            "task_id": task.id,
            "status": task.status,
            "error": str(error),
        }
    )


@shared_task
def download_predict_and_save(task_type="DAILY"):
    if task_type == "DAILY":
        # 获取当前日期的前一天作为结束日期
        end_date = datetime.date.today() - relativedelta(days=2)

        # 开始日期是结束日期的13天之前
        start_date = end_date - relativedelta(days=13)
    elif task_type == "MONTHLY":
        # 获取当前日期的前一天作为结束日期
        end_date = datetime.date.today() - relativedelta(months=1)
        end_date.replace(day=1)

        # 开始日期是结束日期的11个月之前
        start_date = end_date - relativedelta(months=11)
        start_date.replace(day=1)
    else:
        raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    error = ""

    # 创建数据库任务记录
    task = DownloadPredictTask.objects.create(
        start_date=start_date,
        end_date=end_date,
        task_type=task_type,
        source="SCHEDULED",
        status="IN_PROGRESS",
    )
    try:
        # 调用下载、预测和保存函数
        task.input_files, task.input_times = _download_and_save(
            start_date_str, end_date_str, task_type
        )
        task.result_urls = _predict_and_save(
            task.input_files, task.input_times, task_type
        )
        task.status = "COMPLETED"
    except Exception as e:
        task.status = "FAILED"
        error = e
    finally:
        task.save()

    return json.dumps(
        {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "input_files": task.input_files,
            "input_times": task.input_times,
            "result_urls": task.result_urls,
            "task_id": task.id,
            "status": task.status,
            "error": str(error),
        }
    )


@shared_task
def download_predict_and_save_globe(task_type="DAILY"):
    if task_type == "DAILY":
        # 获取当前日期的前一天作为结束日期
        end_date = datetime.date.today() - relativedelta(days=2)

        # 开始日期是结束日期的13天之前
        start_date = end_date - relativedelta(days=13)
    elif task_type == "MONTHLY":
        # 获取当前日期的前一天作为结束日期
        end_date = datetime.date.today() - relativedelta(months=1)
        end_date.replace(day=1)

        # 开始日期是结束日期的11个月之前
        start_date = end_date - relativedelta(months=11)
        start_date.replace(day=1)
    else:
        raise ValueError("task_type must be either 'DAILY' or 'MONTHLY'")
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    error = ""

    # 创建数据库任务记录
    task = DownloadPredictGlobeTask.objects.create(
        start_date=start_date,
        end_date=end_date,
        task_type=task_type,
        source="SCHEDULED",
        status="IN_PROGRESS",
    )
    try:
        # 调用下载、预测和保存函数
        task.input_files, task.input_times = _download_and_save(
            start_date_str, end_date_str, task_type
        )
        task.result_urls = _predict_and_save_globe(
            task.input_files, task.input_times, task_type
        )
        task.status = "COMPLETED"
    except Exception as e:
        task.status = "FAILED"
        error = e
    finally:
        task.save()

    return json.dumps(
        {
            "start_date": start_date_str,
            "end_date": end_date_str,
            "input_files": task.input_files,
            "input_times": task.input_times,
            "result_urls": task.result_urls,
            "task_id": task.id,
            "status": task.status,
            "error": str(error),
        }
    )


if __name__ == "__main__":
    # 获取当前日期的前一天作为结束日期
    end_date = datetime.datetime.now() - datetime.timedelta(days=1)
    end_date_str = end_date.strftime("%Y%m%d")

    # 开始日期是结束日期的12个月之前
    start_date = end_date - datetime.timedelta(days=14)
    start_date_str = start_date.strftime("%Y%m%d")
    # input_files, result_urls = _download_predict_and_save(start_date_str, end_date_str, 'DAILY')
    # print(input_files)
    # print(result_urls)
