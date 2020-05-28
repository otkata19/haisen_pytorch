from django import forms
from .models import Haisen

class HaisenForm(forms.ModelForm):
   """
   新規データ登録画面用のフォーム定義
   """
   class Meta:
       model = Haisen
       fields =['money',]
