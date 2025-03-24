import hashlib
import json
from datetime import datetime
from pathlib import Path

from celery.result import AsyncResult
from dateutil import relativedelta
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from sea_ice_backend import settings
from seaice.models import DownloadPredictTask, DynamicGradTask, ModelInterpreterTask
from seaice.tasks import grad_day_and_return, predict_and_return, grad_and_return


def get_celery_task_result(task_id):
    result = AsyncResult(task_id)
    if result.ready():
        return result.get()
    else:
        return None


@require_POST
@csrf_exempt
def create_day_prediction_task(request):
    days = 14
    try:
        data = json.loads(request.body).get("data")
        start_date_str = data.get("start_date")
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        image_paths = data.get("image_paths", [])

        if len(image_paths) != days:
            return JsonResponse(
                {
                    "error": f"Please provide exactly {days} image paths for daily prediction"
                },
                status=400,
            )

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

        return JsonResponse(
            {"data": {"task_id": task.id, "celery_id": async_result.id}}
        )

    except Exception as e:
        print(e)
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def get_day_prediction_result(request, task_id):
    try:
        task = DownloadPredictTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return JsonResponse({"error": "Task is not yet completed"}, status=400)
        elif task.status == "FAILED":
            return JsonResponse({"error": "Task failed"}, status=500)
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

        return JsonResponse({"data": data})

    except DownloadPredictTask.DoesNotExist:
        return JsonResponse({"error": "Task not found"}, status=404)
    except Exception as e:
        print(e)
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
@csrf_exempt
def create_month_prediction_task(request):
    months = 12
    try:
        data = json.loads(request.body).get("data")
        start_date_str = data.get("start_date")
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        image_paths = data.get("image_paths", [])

        if len(image_paths) != months:
            return JsonResponse(
                {
                    "error": f"Please provide exactly {months} image paths for monthly prediction"
                },
                status=400,
            )
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

        return JsonResponse(
            {"data": {"task_id": task.id, "celery_id": async_result.id}}
        )

    except Exception as e:
        print(e)
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def get_month_prediction_result(request, task_id):
    try:
        task = DownloadPredictTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return JsonResponse({"error": "Task is not yet completed"}, status=400)
        elif task.status == "FAILED":
            return JsonResponse({"error": "Task failed"}, status=500)
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

        return JsonResponse({"data": data})

    except DownloadPredictTask.DoesNotExist:
        return JsonResponse({"error": "Task not found"}, status=404)
    except Exception as e:
        print(e)
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
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
            return JsonResponse(
                {"error": "No completed daily prediction task found"}, status=404
            )

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

        return JsonResponse({"data": data})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
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
            return JsonResponse(
                {"error": "No completed monthly prediction task found"}, status=404
            )

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

        return JsonResponse({"data": data})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
@csrf_exempt
def upload_image(request):
    if "file" not in request.FILES:
        return JsonResponse({"error": "No image file provided"}, status=400)

    image = request.FILES["file"]
    if not image.name.lower().endswith(".png"):
        return JsonResponse(
            {"error": "Invalid image format. Only PNG is allowed."}, status=400
        )

    # Compute the hash of the file content (using SHA256)
    image_hash = hashlib.sha256(image.read()).hexdigest()

    # Generate the filename based on the hash
    unique_filename = f"{image_hash}.png"

    # Define the upload path using pathlib
    upload_path = Path("uploads") / unique_filename

    # Check if file already exists (if it does, we do not need to save it again)
    if default_storage.exists(str(upload_path)):
        image_url = Path(settings.MEDIA_ROOT) / upload_path
        return JsonResponse(
            {"message": "Image already exists", "image_url": str(image_url)}, status=200
        )

    # Reset the file pointer to the beginning, as reading the file consumes the stream
    image.seek(0)

    # Save the uploaded file to the specified path
    saved_path = default_storage.save(str(upload_path), ContentFile(image.read()))

    # Construct the image URL using pathlib
    image_url = Path(settings.MEDIA_ROOT) / saved_path

    return JsonResponse(
        {"message": "Image uploaded successfully", "image_url": str(image_url)}
    )


@require_POST
@csrf_exempt
def create_dynamics_analysis(request):
    try:
        data = json.loads(request.body).get("data")

        start_time_str = data.get("start_time")
        start_time = datetime.strptime(start_time_str, "%Y%m")
        end_time_str = data.get("end_time")
        end_time = datetime.strptime(end_time_str, "%Y%m")
        grad_month = data.get("grad_month")
        grad_type = data.get("grad_type")

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

        return JsonResponse(
            {"data": {"task_id": task.id, "celery_id": async_result.id}}
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def get_dynamics_analysis_result(request, task_id):
    try:
        task = DynamicGradTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return JsonResponse({"error": "Task is not yet completed"}, status=400)
        elif task.status == "FAILED":
            return JsonResponse({"error": "Task failed"}, status=500)
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

        return JsonResponse({"data": data})

    except DynamicGradTask.DoesNotExist:
        return JsonResponse({"error": "Task not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_POST
@csrf_exempt
def create_model_interpreter(request):
    try:
        data = json.loads(request.body).get("data")

        start_time_str = data.get("start_time")
        start_time = datetime.strptime(start_time_str, "%Y%m%d")
        end_time_str = data.get("end_time")
        end_time = datetime.strptime(end_time_str, "%Y%m%d")
        grad_day = data.get("grad_day")
        grad_type = data.get("grad_type")

        # 创建数据库任务记录
        task = ModelInterpreterTask.objects.create(
            start_date=start_time,
            end_date=end_time,
            grad_day=grad_day,
            grad_type=grad_type,
            status="IN_PROGRESS",
        )
        # 保存任务ID
        task.save()

        async_result = grad_day_and_return.delay(
            start_time, end_time, int(grad_day), grad_type, task.id
        )

        return JsonResponse(
            {"data": {"task_id": task.id, "celery_id": async_result.id}}
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def get_model_interpreter_result(request, task_id):
    try:
        task = ModelInterpreterTask.objects.get(id=task_id)

        if task.status == "IN_PROGRESS":
            return JsonResponse({"error": "Task is not yet completed"}, status=400)
        elif task.status == "FAILED":
            return JsonResponse({"error": "Task failed"}, status=500)
        data = [settings.HOST_PREFIX + task.result_urls[0]]

        return JsonResponse({"data": data})

    except ModelInterpreterTask.DoesNotExist:
        return JsonResponse({"error": "Task not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
