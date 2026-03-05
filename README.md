# Gaussian Reconstruction Backend

基于视频的高斯泼溅（Gaussian Splatting）三维重建后端服务。

## 功能特性

- 视频上传和任务管理
- 自动化重建流水线：帧提取 → SfM → 高斯泼溅训练 → 压缩
- 多进程任务队列，支持并发处理
- 任务状态查询（单个/批量）
- SQLite 持久化存储
- 优雅的 Ctrl+C 退出

## 系统要求

- Python 3.10+
- Node.js (用于 splat-transform)
- FFmpeg (CUDA 加速)
- COLMAP
- LichtFeld Studio

## 安装

1. 克隆仓库并安装依赖：

```bash
uv sync
```

2. 复制配置文件并修改：

```bash
copy config.jsonc.example config.jsonc
```

编辑 `config.jsonc`，设置正确的可执行文件路径和参数。

## 配置说明

`config.jsonc` 包含以下配置项：

- `PORT`: 服务端口号（默认 4000）
- `BINARIES`: 可执行文件路径
  - `FFMPEG_PATH`: FFmpeg 路径
  - `LICHTFELD_PATH`: LichtFeld Studio 路径
  - `COLMAP_PATH`: COLMAP 路径
- `STORAGE`: 存储配置
  - `WORK_DIRECTORY`: 工作目录
  - `DATABASE_PATH`: SQLite 数据库路径
- `FRAME_EXTRACTION`: 帧提取参数
- `LICHTFELD_PARAMS`: 训练参数

## 运行

```bash
# 使用默认端口 (config.jsonc 中的 PORT)
uv run python start_api.py

# 指定端口
uv run python start_api.py 8080
```

服务启动后访问：
- API 文档：http://localhost:4000/docs
- ReDoc 文档：http://localhost:4000/redoc

## API 接口

### 上传视频

```
POST /upload
Content-Type: multipart/form-data

参数：
- file: 视频文件

响应：
{
  "task_id": "uuid-string",
  "message": "Task created successfully"
}
```

支持的视频格式：`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`

### 查询单个任务状态

```
GET /status/{task_id}

响应：
{
  "task_id": "uuid-string",
  "status": "waiting|preprocessing|sfm|reconstruction|compress|finish|failure",
  "exists": true
}
```

### 批量查询任务状态

```
POST /status/batch
Content-Type: application/json

请求体：
{
  "task_ids": ["uuid-1", "uuid-2", ...]
}

响应：
{
  "results": [
    {"task_id": "uuid-1", "status": "...", "exists": true},
    {"task_id": "uuid-2", "status": "...", "exists": true}
  ]
}
```

### 队列统计

```
GET /queue/stats

响应：
{
  "total": 5,
  "waiting": 2,
  "preprocessing": 1,
  "sfm": 1,
  "reconstruction": 1,
  "compress": 0
}
```

## 任务状态说明

| 状态 | 说明 |
|------|------|
| `waiting` | 等待处理 |
| `preprocessing` | 帧提取中 |
| `sfm` | SfM 重建中 |
| `reconstruction` | 高斯泼溅训练中 |
| `compress` | 结果压缩中 |
| `finish` | 任务完成 |
| `failure` | 任务失败 |

## 架构设计

### 流水线架构

```
上传视频 → Waiting → Preprocessing → SfM → Reconstruction → Compress → Finish
                          ↓            ↓          ↓             ↓
                       (失败时转到 Failure 状态)
```

### 多进程队列

- 每个阶段（Preprocessing、SfM、Reconstruction、Compress）运行在独立的 worker 进程中
- 使用 `multiprocessing.Queue` 进行进程间通信
- 使用 `multiprocessing.Manager` 共享任务状态
- 每个阶段同时只处理一个任务，确保资源不被过度占用

### 数据持久化

- 进行中的任务存储在内存队列中
- 完成（finish）和失败（failure）的任务存储在 SQLite 数据库中
- 数据库记录包含：任务 ID、状态、创建时间、完成时间

## 项目结构

```
.
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI 路由
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── frame_extraction.py
│   │   ├── colmap_sfm.py
│   │   ├── lichtfeld_recon.py
│   │   └── compress.py
│   ├── config.py              # 配置加载
│   ├── database.py            # SQLite 数据库
│   ├── task_queue.py          # 任务队列管理
│   └── workers.py             # Worker 进程
├── config.jsonc               # 配置文件
├── start_api.py               # 主入口
└── README.md
```

## 开发说明

### 添加新的处理阶段

1. 在 `app/workers.py` 中添加新的 worker 函数
2. 在 `app/task_queue.py` 中添加新的队列
3. 在 `start_api.py` 中启动新的 worker 进程
4. 更新 `TaskStatus` 枚举

### 调试

- 查看 worker 进程输出：每个 worker 会打印处理日志
- 查看 FastAPI 日志：uvicorn 会打印 HTTP 请求日志
- 检查数据库：`sqlite3 output/tasks.db "SELECT * FROM tasks;"`

## 许可证

MIT
