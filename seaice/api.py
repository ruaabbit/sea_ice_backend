import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from celery.result import AsyncResult
from dateutil import relativedelta
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from ninja import NinjaAPI, Schema, File, UploadedFile

from sea_ice_backend import settings
from seaice.models import (
    DownloadPredictTask,
    DynamicGradTask,
    ModelInterpreterTask,
    DownloadPredictGlobeTask,
)
from seaice.tasks import grad_and_return, grad_day_and_return, predict_and_return

api = NinjaAPI(
    title="Sea Ice API",
    description="API for sea ice prediction and analysis",
    version="1.0.0",
)


# Schemas
class StandardResponse(Schema):
    success: bool
    message: str
    status: str = "COMPLETED"  # 默认值
    data: Optional[Dict] = None


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
    x1: Optional[int] = 0
    y1: Optional[int] = 0
    x2: Optional[int] = 432
    y2: Optional[int] = 432


class ModelInterpreterIn(Schema):
    start_time: str
    end_time: str
    pred_gap: int
    grad_type: str
    position: Optional[str] = None
    variable: Optional[int] = None


class ImagePathDate(Schema):
    path: str
    date: str


# Utility functions
def get_celery_task_result(task_id):
    result = AsyncResult(task_id)
    if result.ready():
        return result.get()
    return None


@api.post("/upload/image", response=StandardResponse)
def upload_image(request, file: UploadedFile = File(...)):
    try:
        if not file.name.lower().endswith(".png"):
            return StandardResponse(
                success=False, message="无效的图像格式。仅允许PNG格式", data=None
            )

        file_content = file.read()
        image_hash = hashlib.sha256(file_content).hexdigest()
        unique_filename = f"{image_hash}.png"
        upload_path = Path("uploads") / unique_filename

        if default_storage.exists(str(upload_path)):
            image_url = Path(settings.MEDIA_ROOT) / upload_path
            return StandardResponse(
                success=True, message="图像已存在", data={"image_url": str(image_url)}
            )

        saved_path = default_storage.save(str(upload_path), ContentFile(file_content))
        image_url = Path(settings.MEDIA_ROOT) / saved_path

        return StandardResponse(
            success=True, message="图像上传成功", data={"image_url": str(image_url)}
        )
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.post("/predict/day", response=StandardResponse)
def create_day_prediction_task(request, data: DayPredictionIn):
    days = 14
    try:
        start_date = datetime.strptime(data.start_date, "%Y/%m/%d")
        image_paths = data.image_paths

        if len(image_paths) != days:
            return StandardResponse(
                success=False,
                message=f"请提供正好{days}个图像路径用于逐日预测",
                data=None,
            )

        task = DownloadPredictTask.objects.create(
            start_date=start_date,
            end_date=start_date + relativedelta.relativedelta(days=days),
            task_type="DAILY",
            source="API",
            status="IN_PROGRESS",
        )

        async_result = predict_and_return.delay(image_paths, [], "DAILY", task.id)

        return StandardResponse(
            success=True,
            message="逐日预测任务已创建",
            data={"task_id": task.id, "celery_id": async_result.id},
        )

    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.get("/predict/day/{task_id}", response=StandardResponse)
def get_day_prediction_result(request, task_id: int):
    try:
        task = DownloadPredictTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return StandardResponse(
                success=False,
                message="任务正在处理中，请稍后再试",
                status="IN_PROGRESS",
                data=None,
            )
        elif task.status == "FAILED":
            return StandardResponse(
                success=False,
                message="任务处理失败，请联系管理员",
                status="FAILED",
                data=None,
            )

        images = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(days=i + 1)
            images.append(
                {
                    "path": url,
                    "date": current_date.strftime("%Y-%m-%d"),
                }
            )

        return StandardResponse(
            success=True, message="获取逐日预测结果成功", data={"images": images}
        )

    except DownloadPredictTask.DoesNotExist:
        return StandardResponse(success=False, message="任务未找到", data=None)
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.post("/predict/month", response=StandardResponse)
def create_month_prediction_task(request, data: MonthPredictionIn):
    months = 12
    try:
        start_date = datetime.strptime(data.start_date, "%Y/%m/%d")
        image_paths = data.image_paths

        if len(image_paths) != months:
            return StandardResponse(
                success=False,
                message=f"请提供正好{months}个图像路径用于逐月预测",
                data=None,
            )

        current_date = start_date
        input_times = []
        while current_date < start_date + relativedelta.relativedelta(months=months):
            input_times.append(current_date.month)
            current_date += relativedelta.relativedelta(months=1)

        task = DownloadPredictTask.objects.create(
            start_date=start_date,
            end_date=start_date + relativedelta.relativedelta(months=months),
            task_type="MONTHLY",
            source="API",
            status="IN_PROGRESS",
        )

        async_result = predict_and_return.delay(
            image_paths, input_times, "MONTHLY", task.id
        )

        return StandardResponse(
            success=True,
            message="逐月预测任务已创建",
            data={"task_id": task.id, "celery_id": async_result.id},
        )

    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.get("/predict/month/{task_id}", response=StandardResponse)
def get_month_prediction_result(request, task_id: int):
    try:
        task = DownloadPredictTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return StandardResponse(
                success=False,
                message="任务正在处理中，请稍后再试",
                status="IN_PROGRESS",
                data=None,
            )
        elif task.status == "FAILED":
            return StandardResponse(
                success=False,
                message="任务处理失败，请联系管理员",
                status="FAILED",
                data=None,
            )

        images = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i + 1)
            images.append(
                {
                    "path": url,
                    "date": current_date.strftime("%Y-%m"),
                }
            )

        return StandardResponse(
            success=True, message="获取逐月预测结果成功", data={"images": images}
        )

    except DownloadPredictTask.DoesNotExist:
        return StandardResponse(success=False, message="任务未找到", data=None)
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.get("/realtime/day", response=StandardResponse)
def realtime_day_prediction(request):
    try:
        task = (
            DownloadPredictGlobeTask.objects.filter(
                task_type="DAILY", status="COMPLETED", source="SCHEDULED"
            )
            .order_by("-created_at")
            .first()
        )
        if not task:
            return StandardResponse(
                success=False, message="未找到已完成的逐日预测任务", data=None
            )

        images = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(days=i + 14)
            images.append(
                {
                    "path": url,
                    "date": current_date.strftime("%Y-%m-%d"),
                }
            )

        return StandardResponse(
            success=True, message="获取实时逐日预测成功", data={"images": images}
        )
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.get("/realtime/month", response=StandardResponse)
def realtime_month_prediction(request):
    try:
        task = (
            DownloadPredictGlobeTask.objects.filter(
                task_type="MONTHLY", status="COMPLETED", source="SCHEDULED"
            )
            .order_by("-created_at")
            .first()
        )
        if not task:
            return StandardResponse(
                success=False, message="未找到已完成的逐月预测任务", data=None
            )

        images = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i + 12)
            images.append(
                {
                    "path": url,
                    "date": current_date.strftime("%Y-%m"),
                }
            )

        return StandardResponse(
            success=True, message="获取实时逐月预测成功", data={"images": images}
        )
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.post("/dynamics/analysis", response=StandardResponse)
def create_dynamics_analysis(request, data: DynamicsAnalysisIn):
    try:
        start_time = datetime.strptime(data.start_time, "%Y%m")
        end_time = datetime.strptime(data.end_time, "%Y%m")

        task = DynamicGradTask.objects.create(
            start_date=start_time,
            end_date=end_time,
            grad_month=data.grad_month,
            grad_type=data.grad_type,
            x1=data.x1,
            y1=data.y1,
            x2=data.x2,
            y2=data.y2,
            status="IN_PROGRESS",
        )

        async_result = grad_and_return.delay(
            start_time,
            end_time,
            int(data.grad_month),
            data.grad_type,
            data.x1,
            data.y1,
            data.x2,
            data.y2,
            task.id,
        )

        return StandardResponse(
            success=True,
            message="动力学分析任务已创建",
            data={"task_id": task.id, "celery_id": async_result.id},
        )
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.get("/dynamics/analysis/{task_id}", response=StandardResponse)
def get_dynamics_analysis_result(request, task_id: int):
    try:
        task = DynamicGradTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return StandardResponse(
                success=False,
                message="任务正在处理中，请稍后再试",
                status="IN_PROGRESS",
                data=None,
            )
        elif task.status == "FAILED":
            return StandardResponse(
                success=False,
                message="任务处理失败，请联系管理员",
                status="FAILED",
                data=None,
            )

        images = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i)
            images.append(
                {
                    "path": url,
                    "date": current_date.strftime("%m") + "月",
                }
            )

        return StandardResponse(
            success=True, message="获取动力学分析结果成功", data={"images": images}
        )

    except DynamicGradTask.DoesNotExist:
        return StandardResponse(success=False, message="任务未找到", data=None)
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.post("/model/interpreter", response=StandardResponse)
def create_model_interpreter(request, data: ModelInterpreterIn):
    try:
        start_time = datetime.strptime(data.start_time, "%Y%m%d")
        end_time = datetime.strptime(data.end_time, "%Y%m%d")

        task = ModelInterpreterTask.objects.create(
            start_date=start_time,
            end_date=end_time,
            pred_gap=data.pred_gap,
            grad_type=data.grad_type,
            position=data.position,
            variable=data.variable,
            status="IN_PROGRESS",
        )

        async_result = grad_day_and_return.delay(
            start_time,
            end_time,
            int(data.pred_gap),
            data.grad_type,
            data.position,
            data.variable,
            task.id,
        )

        return StandardResponse(
            success=True,
            message="模型解释任务已创建",
            data={"task_id": task.id, "celery_id": async_result.id},
        )
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)


@api.get("/model/interpreter/{task_id}", response=StandardResponse)
def get_model_interpreter_result(request, task_id: int):
    try:
        task = ModelInterpreterTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return StandardResponse(
                success=False,
                message="任务正在处理中，请稍后再试",
                status="IN_PROGRESS",
                data=None,
            )
        elif task.status == "FAILED":
            return StandardResponse(
                success=False,
                message="任务处理失败，请联系管理员",
                status="FAILED",
                data=None,
            )

        images = task.result_urls

        return StandardResponse(
            success=True, message="获取模型解释结果成功", data={"images": images}
        )

    except ModelInterpreterTask.DoesNotExist:
        return StandardResponse(success=False, message="任务未找到", data=None)
    except Exception as e:
        return StandardResponse(success=False, message=f"发生错误: {str(e)}", data=None)
