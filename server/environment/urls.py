from django.urls import path
from .views import *

urlpatterns = [
    path('', environmental_data, name = "environmental-data"),
]