from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.views.decorators.http import require_GET


def hello_world(request):
    return HttpResponse("Hello World!")


def hello_world_json(request):
    return JsonResponse({'hello': 'world'})


# 日预测视图
@require_GET
def day_prediction(request):
    # 获取前端传递的startDate参数
    start_date_str = request.GET.get('startDate')
    try:
        start_date = datetime.strptime(start_date_str, '%Y/%m/%d')
    except (ValueError, TypeError):
        return JsonResponse({'error': 'Invalid start date'}, status=400)

    days = 7
    images = []

    start_date_fake = datetime.strptime('2019/09/15', '%Y/%m/%d')

    # 生成7天的图片路径和日期信息
    for i in range(days):
        current_date = start_date + timedelta(days=i)
        current_date_fake = start_date_fake + timedelta(days=i)
        # 假设图片路径以日期为名，并存储在某个路径下
        path = f"picture/arctic-sea-ice/{current_date_fake.year}0915-{current_date_fake.year}0928/{current_date_fake.strftime('%Y%m%d')}.png"
        images.append({
            'path': path,
            'date': current_date.strftime('%Y-%m-%d')
        })

    return JsonResponse({'data': images})


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
