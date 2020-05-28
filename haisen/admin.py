from django.contrib import admin
# Register your models here.
#ここから下を追加
from .models import Category, Haisen

#追加
class HaisenAdmin(admin.ModelAdmin):
    list_display=('money',)

admin.site.register(Category)
admin.site.register(Haisen,HaisenAdmin)
