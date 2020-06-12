from django.urls import path

from . import views

app_name = 'haisen'

urlpatterns = [
    path('haisen_list/', views.HaisenListView.as_view(), name='haisen_list'),
    path('haisen_create/', views.HaisenCreateView.as_view(), name='haisen_create'),
    path('create_done/', views.create_done, name='create_done'),
    ]
