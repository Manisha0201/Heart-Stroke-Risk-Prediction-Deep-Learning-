"""Heart URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Heart import views as view
from users import views as users
from admins import views as admins


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', view.index, name='index'),
    path('logout/', view.logout, name='logout'),
    path('generate_pdf/<str:risk_level>/<str:risk_percentage>/<str:caused_features>/<str:recommendation>/', users.generate_pdf, name='generate_pdf'),



    path('UserLogin/', users.UserLogin, name='UserLogin'),
    path('UserRegisterAction/', users.UserRegisterAction, name='UserRegisterAction'),
    path('UserLoginCheck/', users.UserLoginCheck, name='UserLoginCheck'),
    path('UserDataView/', users.UserDataView, name='UserDataView'),
    path('UserMachineLearning/', users.UserMachineLearning, name='UserMachineLearning'),


    path('AdminLogin/', admins.AdminLogin, name='AdminLogin'),
    path('AdminLoginCheck/', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivaUsers/', admins.ActivaUsers, name='ActivaUsers'),
    path('adminML/', admins.adminML, name='adminML'),
    path('output_prediction/', users.output_prediction, name='output_prediction'),
    path('user_data_prediction/', users.user_data_prediction, name='user_data_prediction'),


]
