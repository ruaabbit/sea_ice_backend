from django.contrib import admin

from .models import DownloadPredictTask


@admin.register(DownloadPredictTask)
class DownloadPredictTaskAdmin(admin.ModelAdmin):
    list_display = ('id', 'start_date', 'end_date', 'task_type', 'status', 'created_at', 'updated_at')
    search_fields = ('task_type', 'status')
    list_filter = ('task_type', 'status', 'created_at', 'updated_at')
    ordering = ('-created_at',)
