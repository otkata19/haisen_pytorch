from django.urls import path

from . import views

app_name = 'haisen'

urlpatterns = [
    path('notes/', views.notes, name='notes'),
    path('haisen_create/', views.HaisenCreateView.as_view(), name='haisen_create'),
    path('haiku_create_done/', views.haiku_create_done, name='haiku_create_done'),
    path('create_done/', views.create_done, name='create_done'),
    path('finished_work/', views.form_post, name='finished_work'),
    path('registration/', views.registration, name='registration'),
    path('haisen_list/', views.haisen_list, name='haisen_list'),
    ]
