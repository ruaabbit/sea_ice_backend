from django.db import models


class DownloadPredictTask(models.Model):
    SOURCE_CHOICES = [
        ("MANUAL", "手动创建"),
        ("SCHEDULED", "定时任务"),
        ("API", "API调用"),
    ]

    start_date = models.DateField()
    end_date = models.DateField()
    status = models.CharField(
        max_length=20,
        choices=[
            ("PENDING", "Pending"),
            ("IN_PROGRESS", "In Progress"),
            ("COMPLETED", "Completed"),
            ("FAILED", "Failed"),
        ],
        default="PENDING",
    )
    input_files = models.JSONField(default=list, blank=True)
    result_urls = models.JSONField(default=list, blank=True)
    input_times = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    task_type = models.CharField(
        max_length=20,
        choices=[("DAILY", "Daily"), ("MONTHLY", "Monthly")],
        default="DAILY",
    )
    source = models.CharField(
        max_length=50,
        choices=SOURCE_CHOICES,
        default="MANUAL",
        help_text="标识任务的创建来源",
    )

    def __str__(self):
        return f"Download Predict Task from {self.start_date} to {self.end_date} - {self.status}"

    class Meta:
        verbose_name = "下载预测任务"
        verbose_name_plural = verbose_name


class DownloadPredictGlobeTask(models.Model):
    SOURCE_CHOICES = [
        ("MANUAL", "手动创建"),
        ("SCHEDULED", "定时任务"),
        ("API", "API调用"),
    ]

    start_date = models.DateField()
    end_date = models.DateField()
    status = models.CharField(
        max_length=20,
        choices=[
            ("PENDING", "Pending"),
            ("IN_PROGRESS", "In Progress"),
            ("COMPLETED", "Completed"),
            ("FAILED", "Failed"),
        ],
        default="PENDING",
    )
    input_files = models.JSONField(default=list, blank=True)
    result_urls = models.JSONField(default=list, blank=True)
    input_times = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    task_type = models.CharField(
        max_length=20,
        choices=[("DAILY", "Daily"), ("MONTHLY", "Monthly")],
        default="DAILY",
    )
    source = models.CharField(
        max_length=50,
        choices=SOURCE_CHOICES,
        default="MANUAL",
        help_text="标识任务的创建来源",
    )

    def __str__(self):
        return f"Download Predict Globe Task from {self.start_date} to {self.end_date} - {self.status}"

    class Meta:
        verbose_name = "下载预测地球任务"
        verbose_name_plural = verbose_name


class DynamicGradTask(models.Model):
    status = models.CharField(
        max_length=20,
        choices=[
            ("PENDING", "Pending"),
            ("IN_PROGRESS", "In Progress"),
            ("COMPLETED", "Completed"),
            ("FAILED", "Failed"),
        ],
        default="PENDING",
    )

    start_date = models.DateField()
    end_date = models.DateField()
    grad_month = models.IntegerField()
    grad_type = models.CharField(
        max_length=20,
        choices=[("sum", "海冰面积"), ("sqrt", "海冰变化")],
        default="sum",
    )
    x1 = models.IntegerField(default=0)
    y1 = models.IntegerField(default=0)
    x2 = models.IntegerField(default=432)
    y2 = models.IntegerField(default=432)
    result_urls = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Dynamic Grad Task from {self.start_date} to {self.end_date} - {self.status}"

    class Meta:
        verbose_name = "动力学分析任务"
        verbose_name_plural = verbose_name


class ModelInterpreterTask(models.Model):
    status = models.CharField(
        max_length=20,
        choices=[
            ("PENDING", "Pending"),
            ("IN_PROGRESS", "In Progress"),
            ("COMPLETED", "Completed"),
            ("FAILED", "Failed"),
        ],
        default="PENDING",
    )

    start_date = models.DateField()
    end_date = models.DateField()
    pred_gap = models.IntegerField()
    grad_type = models.CharField(
        max_length=20, choices=[("sum", "海冰面积"), ("l2", "L2范数")], default="sum"
    )
    position = models.CharField(max_length=100, blank=True, null=True)
    variable = models.IntegerField(blank=True, null=True)
    result_urls = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Model Interpreter Task from {self.start_date} to {self.end_date} - {self.status}"

    class Meta:
        verbose_name = "模型可解释性任务"
        verbose_name_plural = verbose_name
