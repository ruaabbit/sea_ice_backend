from django.db import models


class DownloadPredictTask(models.Model):
    start_date = models.DateField()
    end_date = models.DateField()
    status = models.CharField(max_length=20, choices=[('PENDING', 'Pending'), ('IN_PROGRESS', 'In Progress'),
                                                      ('COMPLETED', 'Completed'), ('FAILED', 'Failed')],
                              default='PENDING')
    input_files = models.JSONField(default=list, blank=True)
    result_urls = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    task_type = models.CharField(max_length=20, choices=[('DAILY', 'Daily'), ('MONTHLY', 'Monthly')],
                                 default='DAILY')

    def __str__(self):
        return f"Task from {self.start_date} to {self.end_date} - {self.status}"
