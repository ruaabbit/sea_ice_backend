from django.urls import path

from seaice import views

urlpatterns = [
    path('png-upload', views.upload_image),
    path('day-prediction', views.day_prediction),
    path('month-prediction', views.month_prediction),
    path('realtime-day-prediction', views.realtime_day_prediction),
    path('realtime-month-prediction', views.realtime_month_prediction),
    path('dynamics-analysis', views.dynamics_analysis),
]
