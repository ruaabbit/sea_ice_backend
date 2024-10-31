import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from datetime import datetime, timedelta
from django.views.decorators.http import require_GET, require_POST

from sea_ice_backend import settings
from seaice.osi_saf import predict


# 日预测视图
@require_POST
def day_prediction(request):
    days = 14
    try:
        data = json.loads(request.body).get('data')
        start_date_str = data.get('start_date')
        start_date = datetime.strptime(start_date_str, '%Y/%m/%d')
        image_paths = data.get('image_paths', [])

        if len(image_paths) != 14:
            return JsonResponse({'error': 'Please provide exactly 14 image paths'}, status=400)

        # Open images from local paths
        images = []
        for path_str in image_paths:
            path = Path(path_str)
            if path.exists():
                img = Image.open(path)
                images.append(img)
            else:
                return JsonResponse({'error': f'File not found: {path_str}'}, status=400)

        # Generate predictions
        predictions = predict.predict_ice_concentration(images)

        # Save predictions as images and generate URLs
        urls = []
        for i, prediction in enumerate(predictions):
            pred_image = Image.fromarray(np.array((prediction[0] * 255)).astype(np.uint8))
            buffer = io.BytesIO()
            pred_image.save(buffer, format='PNG')
            image_hash = hashlib.sha256(buffer.read()).hexdigest()
            file_name = Path('predicts') / f'{image_hash}.png'
            if not default_storage.exists(str(file_name)):
                file_name = default_storage.save(file_name, ContentFile(buffer.getvalue()))
            file_url = default_storage.url(file_name)
            urls.append(file_url)

        # 生成14天的图片路径和日期信息
        data = []
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            # 假设图片路径以日期为名，并存储在某个路径下
            data.append({
                'path': 'https://seaice.52lxy.one:20443' + urls[i],
                'date': current_date.strftime('%Y-%m-%d')
            })

        return JsonResponse({'data': data})

    except Exception as e:
        print(e)
        return JsonResponse({'error': str(e)}, status=500)


# 月预测视图
@require_GET
def month_prediction(request):
    # 获取startYear和startMonth参数
    start_year = int(request.GET.get('startYear', 0))
    start_month = int(request.GET.get('startMonth', 0))

    if start_year <= 0 or start_month not in range(1, 13):
        return JsonResponse({'error': 'Invalid year or month'}, status=400)

    months = 6
    images = []

    start_year_fake = 2019
    start_month_fake = 1

    # 生成6个月的图片路径和日期信息
    for i in range(months):
        current_month = (start_month + i - 1) % 12 + 1
        current_year = start_year + (start_month + i - 1) // 12
        current_month_fake = (start_month_fake + i - 1) % 12 + 1
        current_year_fake = start_year_fake + (start_month_fake + i - 1) // 12
        # 假设图片路径以月份为名
        path = f"picture/arctic-sea-ice/{current_year_fake}01-{current_year_fake}12/{current_year_fake}{str(current_month_fake).zfill(2)}.png"
        images.append({
            'path': path,
            'date': f"{current_year}-{str(current_month).zfill(2)}"
        })

    return JsonResponse({'data': images})


@require_POST
def upload_image(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)

    image = request.FILES['file']
    if not image.name.lower().endswith('.png'):
        return JsonResponse({'error': 'Invalid image format. Only PNG is allowed.'}, status=400)

    # Compute the hash of the file content (using SHA256)
    image_hash = hashlib.sha256(image.read()).hexdigest()

    # Generate the filename based on the hash
    unique_filename = f"{image_hash}.png"

    # Define the upload path using pathlib
    upload_path = Path('uploads') / unique_filename

    # Check if file already exists (if it does, we do not need to save it again)
    if default_storage.exists(str(upload_path)):
        image_url = Path(settings.MEDIA_URL) / upload_path
        return JsonResponse({'message': 'Image already exists', 'image_url': str(image_url)}, status=200)

    # Reset the file pointer to the beginning, as reading the file consumes the stream
    image.seek(0)

    # Save the uploaded file to the specified path
    saved_path = default_storage.save(str(upload_path), ContentFile(image.read()))

    # Construct the image URL using pathlib
    image_url = Path(settings.MEDIA_URL) / saved_path

    return JsonResponse({'message': 'Image uploaded successfully', 'image_url': str(image_url)})
