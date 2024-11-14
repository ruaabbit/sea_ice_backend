import hashlib
import io
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from dateutil import relativedelta
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from sea_ice_backend import settings
from seaice.models import DownloadPredictTask
from seaice.osi_450_a import predict as predict_month
from seaice.osi_saf import predict as predict_day


# 日预测视图
@require_POST
@csrf_exempt
def day_prediction(request):
    days = 14
    try:
        data = json.loads(request.body).get("data")
        start_date_str = data.get("start_date")
        start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
        image_paths = data.get("image_paths", [])

        if len(image_paths) != 14:
            return JsonResponse(
                {"error": "Please provide exactly 14 image paths"}, status=400
            )

        # Open images from local paths
        images = []
        for path_str in image_paths:
            try:
                with Image.open(path_str) as img:
                    images.append(img.copy())
            except Exception as e:
                return JsonResponse(
                    {"error": f"Failed to open image {path_str}: {str(e)}"}, status=400
                )

        # 创建数据库任务记录
        task = DownloadPredictTask.objects.create(
            start_date=start_date,
            end_date=start_date + timedelta(days=days - 1),
            task_type='DAILY',
            source='API',
            status='IN_PROGRESS'
        )

        # Generate predictions
        predictions = predict_day.predict_ice_concentration_from_images(images)

        # Save predictions as images and generate URLs
        urls = []
        for i, prediction in enumerate(predictions):
            # 生成带时间戳的随机文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_num = random.randint(1000, 9999)
            file_name = f"predict_{timestamp}_{random_num}.png"

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
            urls.append(file_url)
            buffer.close()
        task.result_urls = urls
        task.status = 'COMPLETED'
        task.save()
        # 生成14天的图片路径和日期信息
        data = []
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            # 假设图片路径以日期为名，并存储在某个路径下
            data.append(
                {
                    "path": settings.HOST_PREFIX + urls[i],
                    "date": current_date.strftime("%Y-%m-%d"),
                }
            )
        return JsonResponse({"data": data})

    except Exception as e:
        print(e)
        return JsonResponse({"error": str(e)}, status=500)


# 月预测视图
@require_POST
@csrf_exempt
def month_prediction(request):
    months = 12
    # try:
    data = json.loads(request.body).get("data")
    start_date_str = data.get("start_date")
    start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
    image_paths = data.get("image_paths", [])

    if len(image_paths) != 12:
        return JsonResponse(
            {"error": "Please provide exactly 12 image path for monthly prediction"},
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
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    # Open images from local paths
    images = []
    for path_str in image_paths:
        try:
            with Image.open(path_str) as img:
                images.append(img.copy())
        except Exception as e:
            return JsonResponse(
                {"error": f"Failed to open image {path_str}: {str(e)}"}, status=400
            )

    # 创建数据库任务记录
    task = DownloadPredictTask.objects.create(
        start_date=start_date,
        end_date=start_date + timedelta(days=365),
        task_type='DAILY',
        source='API',
        status='IN_PROGRESS'
    )

    # Generate predictions for the next 12 months
    predictions = predict_month.predict_ice_concentration_from_images(
        images, input_times
    )  # Repeating the same image for each month

    # Save predictions as images and generate URLs
    urls = []
    for i, prediction in enumerate(predictions):
        # Generate a timestamp and random file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_num = random.randint(1000, 9999)
        file_name = f"predict_{timestamp}_{random_num}.png"

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

        # Save the file
        file_path = Path("predicts") / file_name
        if not default_storage.exists(str(file_path)):
            file_path = default_storage.save(
                file_path, ContentFile(buffer.getvalue())
            )
        file_url = default_storage.url(file_path)
        urls.append(file_url)
        buffer.close()
    task.result_urls = urls
    task.status = 'COMPLETED'
    task.save()
    # Generate 12 months of image paths and dates
    data = []
    for i in range(months):
        current_date = start_date + relativedelta.relativedelta(months=i + 1)
        data.append(
            {
                "path": settings.HOST_PREFIX + urls[i],
                "date": current_date.strftime("%Y-%m"),
            }
        )

    return JsonResponse({"data": data})

    # except Exception as e:
    #     print(e)
    #     return JsonResponse({"error": str(e)}, status=500)


@require_GET
def realtime_day_prediction(request):
    try:
        task = DownloadPredictTask.objects.filter(task_type='DAILY', status='COMPLETED', source='SCHEDULED').order_by(
            '-created_at').first()
        if not task:
            return JsonResponse({"error": "No completed daily prediction task found"}, status=404)

        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + timedelta(days=i)
            data.append({
                "path": settings.HOST_PREFIX + url,
                "date": current_date.strftime("%Y-%m-%d"),
            })

        return JsonResponse({"data": data})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def realtime_month_prediction(request):
    try:
        task = DownloadPredictTask.objects.filter(task_type='MONTHLY', status='COMPLETED', source='SCHEDULED').order_by(
            '-created_at').first()
        if not task:
            return JsonResponse({"error": "No completed monthly prediction task found"}, status=404)

        data = []
        start_date = task.start_date
        for i, url in enumerate(task.result_urls):
            current_date = start_date + relativedelta.relativedelta(months=i + 12)
            data.append({
                "path": settings.HOST_PREFIX + url,
                "date": current_date.strftime("%Y-%m"),
            })

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
