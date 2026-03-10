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
- task_id: 任务 ID（必需，由客户端提供的 64 位雪花算法生成的 ID，字符串类型）
- file: 视频文件

响应：
{
  "message": "Task created successfully"
}

错误响应：
- 400: task_id 格式无效（必须是数字字符串）
- 409: task_id 已存在
```

说明：
- task_id 必须是客户端使用雪花算法生成的 64 位整数，以字符串形式传递
- task_id 必须唯一，如果已存在会返回 409 错误
- 支持的视频格式：`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`

### 从对象存储创建任务

```
POST /tasks/from-oss
Content-Type: application/json

请求体：
{
  "task_id": "1234567890123456789",
  "video_url": "https://your-storage.com/path/to/video.mp4"
}

响应：
{
  "task_id": "1234567890123456789",
  "message": "Video downloaded and task created successfully",
  "file_size": 12345678
}

错误响应：
- 400: task_id 格式无效、视频下载失败或格式不支持
- 409: task_id 已存在
- 500: 服务器内部错误
```

说明：
- 从对象存储服务（OSS）下载视频并创建重建任务
- `video_url` 必须是可直接访问的视频文件 URL
- task_id 规则与上传接口相同（雪花算法生成的唯一 ID）
- 通过 URL 扩展名自动检测视频格式
- 如果无法检测格式，默认使用 `.mp4`
- 支持的视频格式与上传接口相同
- 下载超时时间为 30 秒（连接超时）
- 下载失败时会自动清理已创建的工作目录

### 查询单个任务状态

```
GET /status/{task_id}

响应：
{
  "task_id": "1234567890123456789",
  "status": "waiting|preprocessing|sfm|reconstruction|compress|finish|failure",
  "exists": true,
  "completed_at": "2026-03-10T14:30:05Z",  // 仅当 status 为 finish 或 failure 时存在
  "intrinsic_matrix": [fx, 0, cx, 0, fy, cy, 0, 0, 1],  // 仅当 status 为 finish 时存在
  "extrinsic_matrix": [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]  // 仅当 status 为 finish 时存在
}
```

说明：
- `completed_at`: 任务完成时间（UTC 时间，ISO 8601 格式，如 `2026-03-10T14:30:05Z`），仅当任务状态为 `finish` 或 `failure` 时返回
- `intrinsic_matrix`: 相机内参矩阵（3x3 展平为 9 个元素），仅当任务状态为 `finish` 时返回
- `extrinsic_matrix`: 相机外参矩阵（4x4 展平为 16 个元素，世界到相机变换），仅当任务状态为 `finish` 时返回
- 进行中的任务（waiting, preprocessing, sfm, reconstruction, compress）这些字段为 `null`

### 批量查询任务状态

```
POST /status/batch
Content-Type: application/json

请求体：
{
  "task_ids": ["1234567890123456789", "1234567890123456790", ...]
}

响应：
{
  "results": [
    {
      "task_id": "1234567890123456789",
      "status": "finish",
      "exists": true,
      "completed_at": "2026-03-10T14:30:05Z",
      "intrinsic_matrix": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
      "extrinsic_matrix": [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]
    },
    {
      "task_id": "1234567890123456790",
      "status": "preprocessing",
      "exists": true,
      "completed_at": null,
      "intrinsic_matrix": null,
      "extrinsic_matrix": null
    }
  ]
}
```

说明：
- `completed_at`: 任务完成时间（UTC 时间，ISO 8601 格式，如 `2026-03-10T14:30:05Z`），仅当任务状态为 `finish` 或 `failure` 时返回
- `intrinsic_matrix`: 相机内参矩阵（3x3 展平为 9 个元素），仅当任务状态为 `finish` 时返回
- `extrinsic_matrix`: 相机外参矩阵（4x4 展平为 16 个元素，世界到相机变换），仅当任务状态为 `finish` 时返回
- 进行中的任务这些字段为 `null`

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

### 下载重建结果

```
GET /tasks/{task_id}/assets

响应：
- 成功：返回 SOG 文件（application/octet-stream）
- 失败：
  - 404: 任务不存在或文件未找到
  - 400: 任务未完成或失败
```

说明：下载已完成任务的高斯泼溅 SOG 文件。只有状态为 `finish` 的任务才能下载。

### 获取相机元数据

```
GET /tasks/{task_id}/metadata

响应：
{
  "task_id": "1234567890123456789",
  "intrinsic_matrix": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
  "extrinsic_matrix": [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1]
}
```

说明：
- 获取相机内参和外参矩阵，用于点云渲染
- `intrinsic_matrix`: 3x3 内参矩阵，行优先展平为 9 个元素
- `extrinsic_matrix`: 4x4 外参矩阵（世界到相机坐标的变换，即 View Matrix / w2c），行优先展平为 16 个元素
  - 注意：这是 COLMAP 直接输出的世界到相机变换矩阵（w2c），坐标转换在渲染阶段处理
  - 选取的是 image_id 最小的帧（通常是第一帧）的相机位姿
- 元数据在 SfM 阶段完成后可用
- 如果任务还在 waiting 或 preprocessing 阶段，会返回 400 错误

### 上传任务到对象存储

```
POST /tasks/{task_id}/upload-to-oss
Content-Type: application/json

请求体：
{
  "upload_url": "https://your-storage.com/presigned-url"
}

响应：
{
  "message": "File uploaded successfully and local files cleaned up",
  "file_size": 12345678
}

错误响应：
- 400: 任务未完成或失败
- 404: 任务不存在或 SOG 文件未找到
- 500: 上传失败
```

说明：
- 将已完成任务的 SOG 文件上传到对象存储服务（OSS）
- `upload_url` 必须是客户端提供的预签名 URL（支持 PUT 请求）
- 只有状态为 `finish` 的任务才能上传
- 上传成功后返回文件大小（字节）
- 使用 PUT 方法上传，Content-Type 为 `application/octet-stream`
- 上传超时时间为 5 分钟
- **上传成功后会自动删除任务的所有本地文件以释放磁盘空间**

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

### 任务 ID 设计

- 任务 ID 由客户端使用雪花算法（Snowflake）生成
- 64 位整数，在 JSON 中以字符串形式传输
- 保证全局唯一性和时间有序性
- 数据库中以 TEXT 类型存储（SQLite 支持任意精度整数）

### 流水线架构

```
客户端生成 task_id (雪花算法) → 上传视频 → Waiting → Preprocessing → SfM → Reconstruction → Compress → Finish
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
│   ├── logger.py              # 彩色日志系统
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