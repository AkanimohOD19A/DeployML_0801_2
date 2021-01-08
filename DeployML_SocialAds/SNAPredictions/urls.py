from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('predict_sna/', views.predict_sna),
    path('predict_sna/result', views.result)
]