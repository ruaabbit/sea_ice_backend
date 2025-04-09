# 海冰预测后端系统

## 项目简介
这是一个基于Django的海冰预测后端系统，集成了多种海冰预测模型，包括OSI-450-A、OSI-SAF和跨模态预测模型。系统提供数据下载、处理、预测和可视化功能。

## 主要功能
- 海冰浓度预测
- 海冰运动预测
- 多模型集成预测
- 数据下载与预处理
- 预测结果可视化

## 技术栈
- Python 3.x
- Django
- Celery (异步任务处理)
- Gunicorn (生产环境部署)
- NumPy/Pandas (数据处理)
- PyTorch (深度学习模型)

## 安装指南

### 环境准备
1. 安装Miniconda/Anaconda
2. 创建conda环境:
   ```bash
   conda env create -f environment.yaml
   conda activate seaice
   ```

### 依赖安装
```bash
pip install -r requirements.txt
```

## 配置说明
1. 复制.env.example为.env并配置:
   ```bash
   cp .env.example .env
   ```
2. 修改.env中的配置项:
   - 数据库连接
   - 模型路径
   - 数据存储路径

## 运行系统

### 开发模式
```bash
python manage.py runserver
```

### 生产模式
```bash
gunicorn --config gunicorn.conf.py sea_ice_backend.wsgi:application
```

### Celery worker
```bash
celery -A sea_ice_backend worker -l info
```

## 模块说明

### 主要模块
- `seaice/osi_450_a`: OSI-450-A海冰预测模型
- `seaice/osi_saf`: OSI-SAF海冰预测模型  
- `seaice/cross_modality`: 跨模态预测模型
- `seaice/common`: 公共数据处理工具

## API文档
系统提供以下主要API端点:

### 预测API
- `POST /api/predict/osi450a`: OSI-450-A模型预测
- `POST /api/predict/osisaf`: OSI-SAF模型预测
- `POST /api/predict/cross`: 跨模态模型预测

### 数据API
- `GET /api/data/download`: 下载海冰数据
- `GET /api/data/visualize`: 可视化海冰数据

## 贡献指南
1. Fork项目仓库
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 许可证
MIT License
