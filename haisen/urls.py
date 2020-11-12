from django.urls import path

from . import views

app_name = 'haisen'

urlpatterns = [
    path('haisen_list/', views.HaisenListView.as_view(), name='haisen_list'),
    path('haiku_create/', views.HaikuCreateView.as_view(), name='haiku_create'),
    path('haisen_create/', views.HaisenCreateView.as_view(), name='haisen_create'),
    path('haiku_create_done/', views.haiku_create_done, name='haiku_create_done'),
    path('create_done/', views.create_done, name='create_done'),
    path('finished_work/', views.form_post, name='finished_work'),
    ]
