# gunicorn.conf.py

# 设置工作进程数
workers = 1

# 绑定的地址和端口
bind = '127.0.0.1:8000'

# 项目的 WSGI 应用
wsgi_app = 'sea_ice_backend.wsgi:application'
