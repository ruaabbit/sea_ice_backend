# Generated by Django 5.0.8 on 2025-03-28 05:33

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('seaice', '0004_dynamicgradtask_x1_dynamicgradtask_x2_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='DownloadPredictGlobeTask',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('start_date', models.DateField()),
                ('end_date', models.DateField()),
                ('status', models.CharField(
                    choices=[('PENDING', 'Pending'), ('IN_PROGRESS', 'In Progress'), ('COMPLETED', 'Completed'),
                             ('FAILED', 'Failed')], default='PENDING', max_length=20)),
                ('input_files', models.JSONField(blank=True, default=list)),
                ('result_urls', models.JSONField(blank=True, default=list)),
                ('input_times', models.JSONField(blank=True, default=list)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('task_type', models.CharField(choices=[('DAILY', 'Daily'), ('MONTHLY', 'Monthly')], default='DAILY',
                                               max_length=20)),
                ('source',
                 models.CharField(choices=[('MANUAL', '手动创建'), ('SCHEDULED', '定时任务'), ('API', 'API调用')],
                                  default='MANUAL', help_text='标识任务的创建来源', max_length=50)),
            ],
            options={
                'verbose_name': '下载预测地球任务',
                'verbose_name_plural': '下载预测地球任务',
            },
        ),
    ]
