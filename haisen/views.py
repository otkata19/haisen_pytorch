from django.shortcuts import render
from . forms import HaisenForm  #forms.pyからhaisenFormをインポート
from django.urls import reverse_lazy
from django.utils import timezone
from django.template.context_processors import csrf

# Create your views here.
#ここから下を追加
from django.views.generic import CreateView, ListView, UpdateView, DeleteView
from .models import Haisen
from . import forms
import logging

logger = logging.getLogger('development')

#一覧表示用のDjango標準ビュー(ListView)を承継して一覧表示用のクラスを定義
class HaisenListView(ListView):
   #利用するモデルを指定
   model = Haisen
   def post(self, request, *args, **kwargs):
        if self.request.POST.get('kami5', None):
            logger.debug("self.request.POST.get() = " + self.request.POST.get('kami5', None))
        if self.request.POST.getlist('title', None):
            logger.debug("self.request.POST.getlist() = " + self.request.POST.getlist('kami5', None)[0])
        return self.get(request, *args, **kwargs)
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
    #テンプレートを設定
    #デフォルトでは「アプリ名/モデル名_form.html」が設定されている
    template_name = "haisen/haisen_form.html"
    #登録処理が正常終了した場合の遷移先を指定
    success_url = reverse_lazy('haisen:create_done')

class HaikuCreateView(CreateView):
    #利用するモデルを指定
    model = Haisen
    #利用するフォームクラス名を指定
    form_class = HaisenForm
    template_name = "haisen/haiku_form.html"
    #登録処理が正常終了した場合の遷移先を指定
    success_url = reverse_lazy('haisen:haiku_create_done')

from haiku_module import haiku_return
from senryu_module import senryu_return

def haiku_create_done(request):
    user_kami5 = request.POST['kami5']
    # naka7_listには、AIが予測する中七の候補が順番に入る
    naka7_list = haiku_return(user_kami5)
    pre1 = user_kami5 + str(' ') + naka7_list[0]
    pre2 = user_kami5 + str(' ') + naka7_list[1]
    pre3 = user_kami5 + str(' ') + naka7_list[2]
    #登録処理が正常終了した場合に呼ばれるテンプレートを指定
    return render(request, 'haisen/create_done.html', {'total1':pre1, 'total2':pre2, 'total3':pre3})

def create_done(request):
    user_kami5 = request.POST['kami5']
    # naka7_listには、AIが予測する中七の候補が順番に入る
    naka7_list = senryu_return(user_kami5)
    pre1 = user_kami5 + str(' ') + naka7_list[0]
    pre2 = user_kami5 + str(' ') + naka7_list[1]
    pre3 = user_kami5 + str(' ') + naka7_list[2]
    #登録処理が正常終了した場合に呼ばれるテンプレートを指定
    return render(request, 'haisen/create_done.html', {'total1':pre1, 'total2':pre2, 'total3':pre3})

def form_post(request):
    kami5_naka7_list = request.POST['haisen'].split()
    kami5, naka7 = kami5_naka7_list[0], kami5_naka7_list[1]
    shimo5 = request.POST['text1']
    work = kami5 + (' ') + naka7 + (' ') + shimo5
    context = {
        'kami5_naka7': request.POST['haisen'],
        'kami5': kami5,
        'naka7':naka7,
        'shimo5': shimo5,
        'work': work
        }
    
    return render(request, 'haisen/finished_work.html', context)

def haisen_list(request):
    if request.method == "POST":
        haisen, name = request.POST['work'], request.POST['name']
        l = Haisen.objects.create(haisen=haisen, name=name)
    data = Haisen.objects.all().order_by("-time")
    context = {
        'haisen': haisen,
        'name': name,
        'data': data
        }
    
    return render(request, 'haisen/haisen_list.html', context)