# 部署指南

## 概述

本文档详细介绍了多模态文件索引器在不同环境下的部署方案，包括本地开发、测试环境、生产环境和容器化部署。

## 系统要求

### 最小系统要求

- **操作系统**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8+
- **内存**: 8GB RAM (推荐16GB+)
- **存储**: 20GB可用空间
- **网络**: 稳定的互联网连接 (用于下载模型)

### 推荐系统配置

- **CPU**: 8核心以上
- **内存**: 32GB RAM
- **GPU**: NVIDIA GPU with 8GB+ VRAM (可选，用于加速)
- **存储**: SSD 100GB+
- **网络**: 千兆网络

### 依赖服务

- **Milvus**: 2.0+ (向量数据库)
- **Docker**: 20.10+ (容器化部署)
- **Docker Compose**: 1.29+ (多容器编排)

## 本地开发环境部署

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd multimodal-file-indexer

# 创建Python虚拟环境
python -m venv .venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 升级pip
pip install --upgrade pip
```

### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装OCR依赖 (可选)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# macOS:
brew install tesseract tesseract-lang

# Windows:
# 下载并安装 Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. 启动Milvus

#### 使用Docker启动单机版Milvus

```bash
# 下载Milvus配置文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动Milvus
docker-compose up -d

# 检查服务状态
docker-compose ps
```

#### 使用Milvus Lite (轻量版)

```bash
# 安装Milvus Lite
pip install milvus-lite

# 在代码中使用
# 无需额外配置，自动使用本地文件存储
```

### 4. 配置系统

```bash
# 复制配置模板
cp config.json.example config.json

# 编辑配置文件
nano config.json
```

基本开发配置：
```json
{
  "milvus": {
    "host": "localhost",
    "port": 19530,
    "collection_name": "dev_multimodal_files"
  },
  "embedding": {
    "multimodal_model": "BAAI/bge-m3",
    "batch_size": 8,
    "use_fp16": false,
    "device": "cpu"
  },
  "processing": {
    "max_concurrent": 5,
    "enable_ocr": true
  },
  "logging": {
    "level": "DEBUG",
    "console_output": true
  }
}
```

### 5. 验证安装

```bash
# 健康检查
python -m multimodal_indexer.cli health-check

# 处理测试文件
python -m multimodal_indexer.cli process-file ./files/test.pdf

# 启动Web界面
python web_ui.py
```

## 生产环境部署

### 1. 服务器准备

#### 系统配置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y python3 python3-pip python3-venv git curl wget

# 安装OCR依赖
sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng

# 安装音频处理依赖
sudo apt install -y ffmpeg libsndfile1

# 创建应用用户
sudo useradd -m -s /bin/bash multimodal
sudo usermod -aG sudo multimodal
```

#### 防火墙配置

```bash
# 开放必要端口
sudo ufw allow 22      # SSH
sudo ufw allow 5000    # Web服务
sudo ufw allow 19530   # Milvus
sudo ufw enable
```

### 2. 应用部署

#### 部署脚本

```bash
#!/bin/bash
# deploy.sh

set -e

APP_DIR="/opt/multimodal-indexer"
APP_USER="multimodal"
PYTHON_VERSION="3.9"

# 创建应用目录
sudo mkdir -p $APP_DIR
sudo chown $APP_USER:$APP_USER $APP_DIR

# 切换到应用用户
sudo -u $APP_USER bash << EOF
cd $APP_DIR

# 克隆代码
git clone <repository-url> .

# 创建虚拟环境
python$PYTHON_VERSION -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 创建必要目录
mkdir -p logs uploads files models

# 设置权限
chmod +x scripts/*.sh
EOF

echo "应用部署完成"
```

#### 配置文件

生产环境配置 (`/opt/multimodal-indexer/config.json`)：

```json
{
  "milvus": {
    "host": "milvus-cluster.internal",
    "port": 19530,
    "collection_name": "prod_multimodal_files",
    "timeout": 60,
    "retry_times": 5
  },
  "embedding": {
    "multimodal_model": "BAAI/bge-m3",
    "batch_size": 16,
    "use_fp16": true,
    "device": "auto",
    "cache_dir": "/opt/multimodal-indexer/models"
  },
  "processing": {
    "max_concurrent": 15,
    "enable_ocr": true,
    "enable_speech_recognition": false,
    "image_quality": 85
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/multimodal-indexer/app.log",
    "max_size": "50MB",
    "backup_count": 10
  },
  "web": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false,
    "upload_folder": "/opt/multimodal-indexer/uploads"
  }
}
```

### 3. 系统服务配置

#### Systemd服务文件

创建 `/etc/systemd/system/multimodal-indexer.service`：

```ini
[Unit]
Description=Multimodal File Indexer
After=network.target milvus.service
Wants=milvus.service

[Service]
Type=simple
User=multimodal
Group=multimodal
WorkingDirectory=/opt/multimodal-indexer
Environment=PATH=/opt/multimodal-indexer/venv/bin
ExecStart=/opt/multimodal-indexer/venv/bin/python web_ui.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=multimodal-indexer

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096

# 安全设置
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/multimodal-indexer/logs /opt/multimodal-indexer/uploads /var/log/multimodal-indexer

[Install]
WantedBy=multi-user.target
```

#### 启动服务

```bash
# 重新加载systemd配置
sudo systemctl daemon-reload

# 启用服务
sudo systemctl enable multimodal-indexer

# 启动服务
sudo systemctl start multimodal-indexer

# 检查状态
sudo systemctl status multimodal-indexer

# 查看日志
sudo journalctl -u multimodal-indexer -f
```

### 4. 反向代理配置

#### Nginx配置

创建 `/etc/nginx/sites-available/multimodal-indexer`：

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # 重定向到HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL证书配置
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # 文件上传大小限制
    client_max_body_size 100M;
    
    # 代理配置
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # 静态文件缓存
    location /static/ {
        alias /opt/multimodal-indexer/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # 健康检查
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:5000/api/health;
    }
}
```

启用配置：

```bash
# 启用站点
sudo ln -s /etc/nginx/sites-available/multimodal-indexer /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重新加载Nginx
sudo systemctl reload nginx
```

## 容器化部署

### 1. Docker镜像构建

#### Dockerfile

```dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p logs uploads files models

# 设置环境变量
ENV PYTHONPATH=/app
ENV MULTIMODAL_LOGGING_LEVEL=INFO

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m multimodal_indexer.cli health-check || exit 1

# 启动命令
CMD ["python", "web_ui.py"]
```

#### 构建镜像

```bash
# 构建镜像
docker build -t multimodal-indexer:latest .

# 查看镜像
docker images multimodal-indexer
```

### 2. Docker Compose部署

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # Milvus向量数据库
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd_data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - minio_data:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus_data:/var/lib/milvus
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

  # 多模态文件索引器
  multimodal-indexer:
    build: .
    container_name: multimodal-indexer
    ports:
      - "5000:5000"
    volumes:
      - ./files:/app/files
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./models:/app/models
      - ./config.json:/app/config.json
    environment:
      - MULTIMODAL_MILVUS_HOST=milvus
      - MULTIMODAL_MILVUS_PORT=19530
      - MULTIMODAL_LOGGING_LEVEL=INFO
    depends_on:
      milvus:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: multimodal-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - multimodal-indexer
    restart: unless-stopped

volumes:
  etcd_data:
  minio_data:
  milvus_data:

networks:
  default:
    name: multimodal-network
```

#### 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f multimodal-indexer

# 停止服务
docker-compose down

# 停止并删除数据
docker-compose down -v
```

### 3. Kubernetes部署

#### 命名空间

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: multimodal-indexer
```

#### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: multimodal-config
  namespace: multimodal-indexer
data:
  config.json: |
    {
      "milvus": {
        "host": "milvus-service",
        "port": 19530,
        "collection_name": "k8s_multimodal_files"
      },
      "embedding": {
        "multimodal_model": "BAAI/bge-m3",
        "batch_size": 16,
        "use_fp16": true,
        "device": "auto"
      },
      "processing": {
        "max_concurrent": 15,
        "enable_ocr": true
      },
      "logging": {
        "level": "INFO",
        "console_output": true
      },
      "web": {
        "host": "0.0.0.0",
        "port": 5000
      }
    }
```

#### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multimodal-indexer
  namespace: multimodal-indexer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: multimodal-indexer
  template:
    metadata:
      labels:
        app: multimodal-indexer
    spec:
      containers:
      - name: multimodal-indexer
        image: multimodal-indexer:latest
        ports:
        - containerPort: 5000
        env:
        - name: MULTIMODAL_MILVUS_HOST
          value: "milvus-service"
        - name: MULTIMODAL_LOGGING_LEVEL
          value: "INFO"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config.json
          subPath: config.json
        - name: models-volume
          mountPath: /app/models
        - name: uploads-volume
          mountPath: /app/uploads
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: multimodal-config
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: uploads-volume
        persistentVolumeClaim:
          claimName: uploads-pvc
```

#### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: multimodal-service
  namespace: multimodal-indexer
spec:
  selector:
    app: multimodal-indexer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: ClusterIP
```

#### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: multimodal-ingress
  namespace: multimodal-indexer
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: 100m
spec:
  tls:
  - hosts:
    - multimodal.example.com
    secretName: multimodal-tls
  rules:
  - host: multimodal.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: multimodal-service
            port:
              number: 80
```

#### 部署到Kubernetes

```bash
# 应用所有配置
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# 查看部署状态
kubectl get pods -n multimodal-indexer
kubectl get services -n multimodal-indexer
kubectl get ingress -n multimodal-indexer

# 查看日志
kubectl logs -f deployment/multimodal-indexer -n multimodal-indexer
```

## 监控和运维

### 1. 健康检查

#### 应用健康检查

```bash
# CLI健康检查
python -m multimodal_indexer.cli health-check

# HTTP健康检查
curl http://localhost:5000/api/health
```

#### 系统监控脚本

```bash
#!/bin/bash
# monitor.sh

LOG_FILE="/var/log/multimodal-monitor.log"

check_service() {
    if systemctl is-active --quiet multimodal-indexer; then
        echo "$(date): Service is running" >> $LOG_FILE
        return 0
    else
        echo "$(date): Service is down, restarting..." >> $LOG_FILE
        systemctl restart multimodal-indexer
        return 1
    fi
}

check_milvus() {
    if curl -s http://localhost:19530/health > /dev/null; then
        echo "$(date): Milvus is healthy" >> $LOG_FILE
        return 0
    else
        echo "$(date): Milvus is unhealthy" >> $LOG_FILE
        return 1
    fi
}

check_disk_space() {
    USAGE=$(df /opt/multimodal-indexer | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ $USAGE -gt 80 ]; then
        echo "$(date): Disk usage is high: ${USAGE}%" >> $LOG_FILE
        # 清理旧日志
        find /opt/multimodal-indexer/logs -name "*.log.*" -mtime +7 -delete
    fi
}

# 执行检查
check_service
check_milvus
check_disk_space
```

### 2. 日志管理

#### Logrotate配置

创建 `/etc/logrotate.d/multimodal-indexer`：

```
/var/log/multimodal-indexer/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 multimodal multimodal
    postrotate
        systemctl reload multimodal-indexer
    endscript
}
```

### 3. 备份策略

#### 数据备份脚本

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/multimodal-indexer"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR/$DATE

# 备份配置文件
cp /opt/multimodal-indexer/config.json $BACKUP_DIR/$DATE/

# 备份Milvus数据 (如果使用本地存储)
if [ -d "/var/lib/milvus" ]; then
    tar -czf $BACKUP_DIR/$DATE/milvus_data.tar.gz -C /var/lib milvus
fi

# 备份上传文件
tar -czf $BACKUP_DIR/$DATE/uploads.tar.gz -C /opt/multimodal-indexer uploads

# 清理旧备份 (保留7天)
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR/$DATE"
```

### 4. 性能优化

#### 系统调优

```bash
# 增加文件描述符限制
echo "multimodal soft nofile 65536" >> /etc/security/limits.conf
echo "multimodal hard nofile 65536" >> /etc/security/limits.conf

# 优化网络参数
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf
sysctl -p
```

#### 应用调优

```json
{
  "embedding": {
    "batch_size": 32,
    "use_fp16": true
  },
  "processing": {
    "max_concurrent": 20
  },
  "milvus": {
    "index_params": {
      "M": 32,
      "efConstruction": 400
    }
  }
}
```

## 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查日志
   sudo journalctl -u multimodal-indexer -n 50
   
   # 检查配置
   python -m multimodal_indexer.cli check-config
   ```

2. **Milvus连接失败**
   ```bash
   # 检查Milvus状态
   docker ps | grep milvus
   
   # 测试连接
   python -m multimodal_indexer.cli test-milvus
   ```

3. **内存不足**
   ```bash
   # 监控内存使用
   htop
   
   # 调整配置
   # 降低 batch_size 和 max_concurrent
   ```

4. **磁盘空间不足**
   ```bash
   # 清理日志
   sudo journalctl --vacuum-time=7d
   
   # 清理模型缓存
   rm -rf /opt/multimodal-indexer/models/.cache
   ```

### 紧急恢复

```bash
# 停止服务
sudo systemctl stop multimodal-indexer

# 恢复配置
sudo cp /backup/multimodal-indexer/latest/config.json /opt/multimodal-indexer/

# 重建虚拟环境
cd /opt/multimodal-indexer
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 重启服务
sudo systemctl start multimodal-indexer
```