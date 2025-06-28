from django.urls import path
from .views import *

urlpatterns = [
    path('register/', register, name='register'),
    path('login/', login, name='login'),
    path('me/', user, name='user'),
    path('activate/<uidb64>/<token>/', activate_user, name='activate-user'),
    path('forgot-password/', forgot_password, name='forgot-password'),
    path('reset-password/<uidb64>/<token>/', reset_password, name='reset_password'),
    path('google-login/', google_login, name='google-login'),
]
