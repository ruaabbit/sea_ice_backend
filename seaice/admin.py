from django.contrib import admin

from .models import DownloadPredictTask, DynamicGradTask, ModelInterpreterTask

admin.site.site_header = "北极海冰预测系统"
admin.site.site_title = "北极海冰预测系统"
admin.site.index_title = "北极海冰预测系统"


@admin.register(DownloadPredictTask)
class DownloadPredictTaskAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "start_date",
        "end_date",
        "task_type",
        "status",
        "created_at",
        "updated_at",
        "source",
    )
    search_fields = ("task_type", "status")
    list_filter = ("task_type", "status", "created_at", "updated_at", "source")
    ordering = ("-created_at",)


@admin.register(DynamicGradTask)
class DynamicGradTaskAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "start_date",
        "end_date",
        "grad_month",
        "grad_type",
        "status",
        "created_at",
        "updated_at",
    )
    search_fields = ("grad_type", "status")
    list_filter = ("grad_type", "status", "created_at", "updated_at")
    ordering = ("-created_at",)


@admin.register(ModelInterpreterTask)
class ModelInterpreterTaskAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "start_date",
        "end_date",
        "grad_day",
        "grad_type",
        "status",
        "created_at",
        "updated_at",
    )
    search_fields = ("grad_type", "status")
    list_filter = ("grad_type", "status", "created_at", "updated_at")
    ordering = ("-created_at",)
