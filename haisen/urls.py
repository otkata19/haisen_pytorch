from django.urls import path

from . import views

app_name = 'haisen'

urlpatterns = [
    path('haisen_list/', views.HaisenListView.as_view(), name='haisen_list'),
    path('haisen_create/', views.HaisenCreateView.as_view(), name='haisen_create'),
    path('create_done/', views.create_done, name='create_done'),
    path('update/<int:pk>/', views.HaisenUpdateView.as_view(), name='haisen_update'),
    path('update_done/', views.update_done, name='update_done'),
    path('delete/<int:pk>/', views.HaisenDeleteView.as_view(), name='haisen_delete'),
    path('delete_done/', views.delete_done, name='delete_done'),
    ]
