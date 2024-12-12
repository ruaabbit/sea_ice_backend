from django.urls import path

from seaice import views

urlpatterns = [
    path('png-upload', views.upload_image),
    path('day-prediction', views.create_day_prediction_task),
    path('day-prediction/<int:task_id>', views.get_day_prediction_result),
    path('month-prediction', views.create_month_prediction_task),
    path('month-prediction/<int:task_id>', views.get_month_prediction_result),
    path('realtime-day-prediction', views.realtime_day_prediction),
    path('realtime-month-prediction', views.realtime_month_prediction),
    path('dynamics-analysis', views.create_dynamics_analysis),
    path('dynamics-analysis/<int:task_id>', views.get_dynamics_analysis_result),
]
