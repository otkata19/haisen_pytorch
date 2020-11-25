from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from datetime import datetime
from django.utils import timezone

# Create your models here.

class Haisen(models.Model):
   class Meta:
       #テーブル名
       db_table ="Haisen"
       verbose_name ="俳句と川柳"           #追加
       verbose_name_plural ="俳句と川柳"    #追加

   #カラムの定義
   name = models.CharField(verbose_name="作者名", max_length=255)
   haisen = models.CharField(verbose_name="俳句や川柳", max_length=255)
   time = models.DateTimeField(default=timezone.now)
   """
   def __str__(self):
       return self.memo
   """
