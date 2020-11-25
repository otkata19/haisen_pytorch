from django.contrib import admin
# Register your models here.
#ここから下を追加
from .models import Haisen

#追加
class HaisenAdmin(admin.ModelAdmin):
    list_display=('haisen',)

admin.site.register(Haisen,HaisenAdmin)
