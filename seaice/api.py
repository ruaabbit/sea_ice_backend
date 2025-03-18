from datetime import datetime
from typing import List, Optional, Any
import hashlib
from pathlib import Path

from dateutil import relativedelta
from celery.result import AsyncResult
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from ninja import NinjaAPI, Schema, File, UploadedFile
from ninja.responses import Response
from ninja.files import UploadedFile

from sea_ice_backend import settings
from seaice.models import DownloadPredictTask, DynamicGradTask
from seaice.tasks import predict_and_return, grad_and_return


api = NinjaAPI(
    title="Sea Ice API", description="API for sea ice prediction and analysis"
)


# 请求和响应模型
class DayPredictionIn(Schema):
    start_date: str
    image_paths: List[str]


class MonthPredictionIn(Schema):
    start_date: str
    image_paths: List[str]


class DynamicsAnalysisIn(Schema):
    start_time: str
    end_time: str
    grad_month: str
    grad_type: str


class TaskResponse(Schema):
    task_id: int
    celery_id: str


class ImagePathDate(Schema):
    path: str
    date: str


class PredictionResultResponse(Schema):
    data: List[ImagePathDate]


class UploadResponse(Schema):
    message: str
    image_url: str


class ErrorResponse(Schema):
    error: str


def get_celery_task_result(task_id):
    result = AsyncResult(task_id)
    if result.ready():
        return result.get()
    else:
        return None


@api.post(
    "/predict/day", response={200: TaskResponse, 400: ErrorResponse, 500: ErrorResponse}
)
def create_day_prediction_task(request, data: DayPredictionIn):
    days = 14
    try:
        start_date_str = data.start_date
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        image_paths = data.image_paths

        if len(image_paths) != days:
            return 400, {
                "error": f"Please provide exactly {days} image paths for daily prediction"
            }

        # 创建数据库任务记录
        task = DownloadPredictTask.objects.create(
            start_date=start_date,
            end_date=start_date + relativedelta.relativedelta(days=days),
            task_type="DAILY",
            source="API",
            status="IN_PROGRESS",
        )
        # 保存任务ID
        task.save()
        # 异步调用 Celery 任务
        async_result = predict_and_return.delay(image_paths, [], "DAILY", task.id)

        return 200, {"task_id": task.id, "celery_id": async_result.id}

    except Exception as e:
        print(e)
        return 500, {"error": str(e)}


@api.get(
    "/predict/day/{task_id}",
    response={
        200: PredictionResultResponse,
        400: ErrorResponse,
        404: ErrorResponse,
        500: ErrorResponse,
    },
)
def get_day_prediction_result(request, task_id: int):
    try:
        task = DownloadPredictTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return 400, {"error": "Task is not yet completed"}
        elif task.status == "FAILED":
            return 500, {"error": "Task failed"}

        # 生成14天的图片路径和日期信息
        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(days=i + 1)
            data.append(
                {
                    "path": settings.HOST_PREFIX + url,
                    "date": current_date.strftime("%Y-%m-%d"),
                }
            )

        return 200, {"data": data}

    except DownloadPredictTask.DoesNotExist:
        return 404, {"error": "Task not found"}
    except Exception as e:
        print(e)
        return 500, {"error": str(e)}


@api.post(
    "/predict/month",
    response={200: TaskResponse, 400: ErrorResponse, 500: ErrorResponse},
)
def create_month_prediction_task(request, data: MonthPredictionIn):
    months = 12
    try:
        start_date_str = data.start_date
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        image_paths = data.image_paths

        if len(image_paths) != months:
            return 400, {
                "error": f"Please provide exactly {months} image paths for monthly prediction"
            }

        # Initialize the current date to the start date
        current_date = start_date

        # List to store the months
        input_times = []

        # Loop through each month from start date to end date
        while current_date < start_date + relativedelta.relativedelta(months=months):
            input_times.append(current_date.month)
            # Move to the next month
            current_date = current_date + relativedelta.relativedelta(months=1)

        # 创建数据库任务记录
        task = DownloadPredictTask.objects.create(
            start_date=start_date,
            end_date=start_date + relativedelta.relativedelta(months=months),
            task_type="MONTHLY",
            source="API",
            status="IN_PROGRESS",
        )
        # 保存任务ID
        task.save()
        # 异步调用 Celery 任务
        async_result = predict_and_return.delay(
            image_paths, input_times, "MONTHLY", task.id
        )

        return 200, {"task_id": task.id, "celery_id": async_result.id}

    except Exception as e:
        print(e)
        return 500, {"error": str(e)}


@api.get(
    "/predict/month/{task_id}",
    response={
        200: PredictionResultResponse,
        400: ErrorResponse,
        404: ErrorResponse,
        500: ErrorResponse,
    },
)
def get_month_prediction_result(request, task_id: int):
    try:
        task = DownloadPredictTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return 400, {"error": "Task is not yet completed"}
        elif task.status == "FAILED":
            return 500, {"error": "Task failed"}

        # 生成12个月的图片路径和日期信息
        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i + 1)
            data.append(
                {
                    "path": settings.HOST_PREFIX + url,
                    "date": current_date.strftime("%Y-%m"),
                }
            )

        return 200, {"data": data}

    except DownloadPredictTask.DoesNotExist:
        return 404, {"error": "Task not found"}
    except Exception as e:
        print(e)
        return 500, {"error": str(e)}


@api.get(
    "/predict/day/realtime",
    response={200: PredictionResultResponse, 404: ErrorResponse, 500: ErrorResponse},
)
def realtime_day_prediction(request):
    try:
        task = (
            DownloadPredictTask.objects.filter(
                task_type="DAILY", status="COMPLETED", source="SCHEDULED"
            )
            .order_by("-created_at")
            .first()
        )
        if not task:
            return 404, {"error": "No completed daily prediction task found"}

        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(days=i + 14)
            data.append(
                {
                    "path": settings.HOST_PREFIX + url,
                    "date": current_date.strftime("%Y-%m-%d"),
                }
            )

        return 200, {"data": data}
    except Exception as e:
        return 500, {"error": str(e)}


@api.get(
    "/predict/month/realtime",
    response={200: PredictionResultResponse, 404: ErrorResponse, 500: ErrorResponse},
)
def realtime_month_prediction(request):
    try:
        task = (
            DownloadPredictTask.objects.filter(
                task_type="MONTHLY", status="COMPLETED", source="SCHEDULED"
            )
            .order_by("-created_at")
            .first()
        )
        if not task:
            return 404, {"error": "No completed monthly prediction task found"}

        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i + 12)
            data.append(
                {
                    "path": settings.HOST_PREFIX + url,
                    "date": current_date.strftime("%Y-%m"),
                }
            )

        return 200, {"data": data}
    except Exception as e:
        return 500, {"error": str(e)}


@api.post("/upload/image", response={200: UploadResponse, 400: ErrorResponse})
def upload_image(request, file: UploadedFile = File(...)):
    if not file:
        return 400, {"error": "No image file provided"}

    if not file.name.lower().endswith(".png"):
        return 400, {"error": "Invalid image format. Only PNG is allowed."}

    # Compute the hash of the file content (using SHA256)
    file_content = file.read()
    image_hash = hashlib.sha256(file_content).hexdigest()

    # Generate the filename based on the hash
    unique_filename = f"{image_hash}.png"

    # Define the upload path using pathlib
    upload_path = Path("uploads") / unique_filename

    # Check if file already exists (if it does, we do not need to save it again)
    if default_storage.exists(str(upload_path)):
        image_url = Path(settings.MEDIA_ROOT) / upload_path
        return 200, {"message": "Image already exists", "image_url": str(image_url)}

    # Save the uploaded file to the specified path
    saved_path = default_storage.save(str(upload_path), ContentFile(file_content))

    # Construct the image URL using pathlib
    image_url = Path(settings.MEDIA_ROOT) / saved_path

    return 200, {"message": "Image uploaded successfully", "image_url": str(image_url)}


@api.post("/dynamics/analysis", response={200: TaskResponse, 500: ErrorResponse})
def create_dynamics_analysis(request, data: DynamicsAnalysisIn):
    try:
        start_time = datetime.strptime(data.start_time, "%Y%m")
        end_time = datetime.strptime(data.end_time, "%Y%m")
        grad_month = data.grad_month
        grad_type = data.grad_type

        # 创建数据库任务记录
        task = DynamicGradTask.objects.create(
            start_date=start_time,
            end_date=end_time,
            grad_month=grad_month,
            grad_type=grad_type,
            status="IN_PROGRESS",
        )
        # 保存任务ID
        task.save()

        async_result = grad_and_return.delay(
            start_time, end_time, int(grad_month), grad_type, task.id
        )

        return 200, {"task_id": task.id, "celery_id": async_result.id}
    except Exception as e:
        return 500, {"error": str(e)}


@api.get(
    "/dynamics/analysis/{task_id}",
    response={
        200: PredictionResultResponse,
        400: ErrorResponse,
        404: ErrorResponse,
        500: ErrorResponse,
    },
)
def get_dynamics_analysis_result(request, task_id: int):
    try:
        task = DynamicGradTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return 400, {"error": "Task is not yet completed"}
        elif task.status == "FAILED":
            return 500, {"error": "Task failed"}

        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i)
            data.append(
                {
                    "path": settings.HOST_PREFIX + url,
                    "date": current_date.strftime("%m") + "月",
                }
            )

        return 200, {"data": data}

    except DynamicGradTask.DoesNotExist:
        return 404, {"error": "Task not found"}
    except Exception as e:
        return 500, {"error": str(e)}
