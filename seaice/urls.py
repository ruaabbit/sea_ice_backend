from django.urls import path

from seaice import views

urlpatterns = [
    path('helloworld', views.hello_world),
    path('helloworldjson', views.hello_world_json),
    path('day-prediction', views.day_prediction),
    path('month-prediction', views.month_prediction),
]
