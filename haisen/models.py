from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from datetime import datetime
# Create your models here.

class Category(models.Model):
   class Meta:
       #テーブル名の指定
       db_table ="category"
       verbose_name ="カテゴリ"         #追加
       verbose_name_plural ="カテゴリ"  #追加
   #カラム名の定義
   category_name = models.CharField(max_length=255,unique=True)
   def __str__(self):
       return self.category_name

class Haisen(models.Model):
   class Meta:
       #テーブル名
       db_table ="Haisen"
       verbose_name ="俳句と川柳"           #追加
       verbose_name_plural ="俳句と川柳"    #追加

   #カラムの定義
   #date = models.DateField(verbose_name="日付",default=datetime.now)
   #category = models.ForeignKey(Category, on_delete = models.PROTECT, verbose_name="カテゴリ")
   #money = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(10935)], verbose_name="数字", help_text="0から10935までの好きな数字を入れてね")
   money = models.CharField(verbose_name="初句を構成する5音を入力してください", max_length=500)
   """
   def __str__(self):
       return self.memo
   """
