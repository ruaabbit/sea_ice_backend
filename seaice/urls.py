from django.urls import path

from seaice import views

urlpatterns = [
    path('png-upload', views.upload_image),
    path('day-prediction', views.day_prediction),
    path('month-prediction', views.month_prediction),
]
