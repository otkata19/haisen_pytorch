from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from datetime import datetime
# Create your models here.

class Haisen(models.Model):
   class Meta:
       #テーブル名
       db_table ="Haisen"
       verbose_name ="俳句と川柳"           #追加
       verbose_name_plural ="俳句と川柳"    #追加

   #カラムの定義
   money = models.CharField(verbose_name="五・七・五の上五を入力すると、AIが中七の候補を返します。", max_length=500)
   """
   def __str__(self):
       return self.memo
   """
