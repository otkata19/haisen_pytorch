from django.shortcuts import render
from . forms import HaisenForm  #forms.pyからhaisenFormをインポート
from django.urls import reverse_lazy

# Create your views here.
#ここから下を追加
from django.views.generic import CreateView, ListView, UpdateView, DeleteView
from .models import Haisen

#一覧表示用のDjango標準ビュー(ListView)を承継して一覧表示用のクラスを定義
class HaisenListView(ListView):
   #利用するモデルを指定
   model = Haisen
   """
   #データを渡すテンプレートファイルを指定
   template_name = 'haisen/haisen_list.html'

   #家計簿テーブルの全データを取得するメソッドを定義
   def queryset(self):
       return Haisen.objects.all()
   """

class HaisenCreateView(CreateView):
    #利用するモデルを指定
    model = Haisen
    #利用するフォームクラス名を指定
    form_class = HaisenForm
    #登録処理が正常終了した場合の遷移先を指定
    success_url = reverse_lazy('haisen:create_done')

from seq2seq_module import ai_return
def create_done(request):
    haisen_data = Haisen.objects.last()
    user_kami5 = haisen_data.money
    # naka7_listには、AIが予測する中七の候補が順番に入る
    naka7_list = ai_return(user_kami5)
    pre1 = user_kami5 + str('　') + naka7_list[0] + str('　　') + str('〇〇〇〇〇')
    pre2 = user_kami5 + str('　') + naka7_list[1] + str('　') + str('〇〇〇〇〇')
    pre3 = user_kami5 + str('　') + naka7_list[2] + str('　') + str('〇〇〇〇〇')
    #登録処理が正常終了した場合に呼ばれるテンプレートを指定
    return render(request, 'haisen/create_done.html', {'total1':pre1, 'total2':pre2, 'total3':pre3})

class HaisenUpdateView(UpdateView):
   #利用するモデルを指定
   model = Haisen
   #利用するフォームクラス名を指定
   form_class = HaisenForm
   #更新処理が正常終了した場合の遷移先を指定
   success_url = reverse_lazy('haisen:update_done')

def update_done(request):
    #更新処理が正常終了した場合に呼ばれるテンプレートを指定
    return render(request, 'haisen/update_done.html')

class HaisenDeleteView(DeleteView):
    #利用するモデルを指定
    model = Haisen
    #削除処理が正常終了した場合の遷移先を指定
    success_url = reverse_lazy('haisen:delete_done')

def delete_done(request):
    return render(request, 'haisen/delete_done.html')
