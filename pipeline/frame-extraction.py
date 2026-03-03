#!/usr/bin/env python3
"""
Frame Extraction Tool

两阶段帧提取流程：
1. 使用 FFmpeg 提取所有帧到临时目录
2. 使用 sharp-frames 进行锐度评分和筛选，输出到目标目录
"""

import subprocess
import json
import os
import shutil
import tempfile
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import Optional, Union, Dict, Any, List, Callable
from multiprocessing import cpu_count

import cv2
from tqdm import tqdm

from sharp_frames.selection_methods import select_best_n_frames

# Sharpness 计算权重常量
BEST_N_SHARPNESS_WEIGHT = 0.7
BEST_N_DISTRIBUTION_WEIGHT = 0.3


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """使用 OpenCV 获取视频信息"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
    }
    cap.release()
    return info


def _extract_frames_ffmpeg(
    video_path: Path,
    temp_dir: Path,
    resize_factor: int,
    output_format: str,
    ffmpeg_exe: Path,
    log: Callable[[str], None],
) -> int:
    """使用 FFmpeg 提取所有帧到临时目录，返回提取的帧数"""
    output_pattern = str(temp_dir / f"frame_%05d.{output_format}")

    # 构建 FFmpeg 命令（使用 CUDA 硬件加速解码）
    cmd = [
        str(ffmpeg_exe),
        "-hwaccel", "cuda",
        "-i", str(video_path),
        "-fps_mode", "passthrough",  # 保持原始时间戳，避免帧重复
        "-q:v", "2",
        "-hide_banner",
        "-y",
    ]

    # 如果 resize_factor > 1，添加缩放滤镜
    if resize_factor > 1:
        # iw/resize_factor 保持纵横比
        cmd.extend(["-vf", f"scale=iw/{resize_factor}:-2"])

    cmd.append(output_pattern)

    log(f">> FFmpeg: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed with exit code {e.returncode}") from e

    # 统计提取的帧数
    frame_files = list(temp_dir.glob(f"frame_*.{output_format}"))
    return len(frame_files)


def _calculate_sharpness_scores(
    frame_paths: List[str], log: Callable[[str], None]
) -> List[Dict[str, Any]]:
    """计算所有帧的锐度评分"""
    frames_data = []
    num_workers = min(cpu_count(), len(frame_paths)) if frame_paths else 1

    def process_image(path: str) -> float:
        """处理单张图片并返回锐度评分"""
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise RuntimeError(f"Failed to read image: {path}")

        height, width = img_gray.shape
        img_half = cv2.resize(
            img_gray, (width // 2, height // 2), interpolation=cv2.INTER_AREA
        )
        return float(cv2.Laplacian(img_half, cv2.CV_64F).var())

    log("Calculating sharpness scores...")

    import concurrent.futures

    with tqdm(total=len(frame_paths), desc="Scoring frames") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_image, path): (idx, path)
                for idx, path in enumerate(frame_paths)
            }

            for future in concurrent.futures.as_completed(futures):
                idx, path = futures[future]
                try:
                    score = future.result()
                    frames_data.append(
                        {
                            "id": os.path.basename(path),
                            "path": path,
                            "index": idx,
                            "sharpnessScore": score,
                        }
                    )
                except Exception as e:
                    log(f"Warning: {str(e)}")
                progress_bar.update(1)

    frames_data.sort(key=lambda x: x["index"])
    return frames_data


def _select_best_n_frames(
    frames_data: List[Dict[str, Any]],
    num_frames: int,
    min_buffer: int,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    """使用 best-n 方法筛选帧"""
    log(f"Selecting best {num_frames} frames...")

    selected = select_best_n_frames(
        frames_data,
        num_frames,
        min_buffer,
        BEST_N_SHARPNESS_WEIGHT,
        BEST_N_DISTRIBUTION_WEIGHT,
    )

    log(f"Selected {len(selected)} frames")
    return selected


def _save_selected_frames(
    selected_frames: List[Dict[str, Any]],
    output_dir: Path,
    output_format: str,
    log: Callable[[str], None],
) -> int:
    """保存选中的帧到输出目录"""
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Saving {len(selected_frames)} frames to: {output_dir}")

    with tqdm(total=len(selected_frames), desc="Saving frames") as progress_bar:
        for i, frame_data in enumerate(selected_frames):
            src_path = frame_data["path"]
            filename = f"frame_{i + 1:05d}.{output_format}"
            dst_path = output_dir / filename

            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                log(f"Error saving {src_path}: {e}")
                continue

            progress_bar.update(1)

    # 保存元数据
    metadata = {
        "total_selected": len(selected_frames),
        "output_format": output_format,
        "frames": [
            {
                "filename": f"frame_{i + 1:05d}.{output_format}",
                "original_index": f["index"],
                "sharpness_score": f["sharpnessScore"],
            }
            for i, f in enumerate(selected_frames)
        ],
    }

    metadata_path = output_dir / "selected_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return len(selected_frames)


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    ffmpeg_exe: Union[str, Path],
    ratio: float = 0.1,
    min_buffer: int = 3,
    resize_factor: int = 1,
    output_format: str = "jpg",
    keep_temp: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    从视频中提取并筛选帧。

    两阶段流程：
    1. 使用 FFmpeg（CUDA 加速）提取所有帧到临时目录
    2. 计算锐度评分并筛选最佳帧，保存到输出目录

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        ffmpeg_exe: FFmpeg 可执行文件路径（必需）
        ratio: 目标帧比例 0-1（默认 0.1 即 10%）
        min_buffer: 最小帧间隔
        resize_factor: 缩放因子（1=原尺寸，2=长宽各缩小为1/2）
        output_format: 输出格式（jpg 或 png）
        keep_temp: 是否保留临时目录（调试用）
        log: 日志输出函数

    Returns:
        包含提取结果的字典
    """
    video = Path(video_path).resolve()
    output = Path(output_dir).resolve()
    exe = Path(ffmpeg_exe).resolve()
    _log = log or print

    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")

    if not exe.exists():
        raise FileNotFoundError(f"FFmpeg not found: {exe}")

    if not 0 < ratio <= 1:
        raise ValueError(f"ratio must be between 0 and 1, got {ratio}")

    if resize_factor < 1:
        raise ValueError(f"resize_factor must be >= 1, got {resize_factor}")

    # 获取视频信息
    video_info = get_video_info(video)
    total_frames = video_info["frame_count"]

    _log(f"Processing video: {video}")
    _log(f"Video info: {total_frames} frames, {video_info['fps']:.2f} fps, {video_info['duration']:.2f}s")
    _log(f"Output directory: {output}")
    _log(f"Ratio: {ratio:.1%}, Resize factor: {resize_factor}")

    # 计算目标帧数
    num_frames = max(1, int(total_frames * ratio))
    _log(f"Target frames: {num_frames} (from {total_frames} total)")

    temp_dir = None
    try:
        # 阶段 1：使用 FFmpeg 提取所有帧到临时目录
        temp_dir = Path(tempfile.mkdtemp(prefix="frames_"))
        _log(f"Temp directory: {temp_dir}")

        _log("\n=== Phase 1: Extracting all frames with FFmpeg ===")
        extracted_count = _extract_frames_ffmpeg(
            video, temp_dir, resize_factor, output_format, exe, _log
        )
        _log(f"Extracted {extracted_count} frames")

        if extracted_count == 0:
            raise RuntimeError("No frames extracted from video")

        # 获取帧路径列表
        frame_paths = sorted(
            [str(p) for p in temp_dir.glob(f"frame_*.{output_format}")],
            key=lambda x: int(Path(x).stem.split("_")[1]),
        )

        # 阶段 2：计算锐度并筛选
        _log("\n=== Phase 2: Sharpness analysis and selection ===")
        frames_data = _calculate_sharpness_scores(frame_paths, _log)

        if not frames_data:
            raise RuntimeError("No frames could be scored")

        selected_frames = _select_best_n_frames(
            frames_data, num_frames, min_buffer, _log
        )

        # 阶段 3：保存选中的帧
        _log("\n=== Phase 3: Saving selected frames ===")
        saved_count = _save_selected_frames(selected_frames, output, output_format, _log)

        _log(f"\n[SUCCESS] Saved {saved_count} frames to: {output}")

        return {
            "success": True,
            "video_path": str(video),
            "output_dir": str(output),
            "total_frames": total_frames,
            "extracted_count": extracted_count,
            "selected_count": saved_count,
            "ratio": ratio,
            "resize_factor": resize_factor,
            "format": output_format,
        }

    except KeyboardInterrupt:
        _log("\nCancelled by user")
        return {"success": False, "error": "Cancelled by user"}
    except Exception as e:
        _log(f"\n[ERROR] {e}")
        return {"success": False, "error": str(e)}
    finally:
        # 清理临时目录
        if temp_dir and temp_dir.exists() and not keep_temp:
            _log(f"Cleaning up temp directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                _log(f"Warning: Could not clean up temp directory: {e}")


def main():
    """CLI 入口点"""
    parser = ArgumentParser(
        description="Frame Extraction Tool - 提取视频所有帧并进行锐度智能筛选",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取最佳 10% 帧，原尺寸
  %(prog)s video.mp4 output_frames --exe ffmpeg.exe --ratio 0.1

  # 提取最佳 30% 帧，长宽缩小为 1/2
  %(prog)s video.mp4 output --exe ffmpeg.exe --ratio 0.3 -r 2

  # 调试时保留临时文件
  %(prog)s video.mp4 output --exe ffmpeg.exe --keep-temp
        """,
    )

    # 必需参数
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("output", help="输出目录")

    # 基本参数
    parser.add_argument("--exe", required=True, help="FFmpeg 可执行文件路径")
    parser.add_argument(
        "--format", dest="output_format", choices=["jpg", "png"], default="jpg",
        help="输出格式 (默认: jpg)"
    )
    parser.add_argument(
        "-r", type=int, default=1, dest="resize_factor",
        help="缩放因子: 1=原尺寸, 2=长宽各缩小为1/2 (默认: 1)"
    )
    parser.add_argument("--keep-temp", action="store_true", help="保留临时目录（调试用）")

    # best-n 参数
    selection_group = parser.add_argument_group("筛选参数")
    selection_group.add_argument(
        "--ratio", type=float, default=0.2,
        help="目标帧比例 0-1 (默认: 0.2 即 20%)"
    )
    selection_group.add_argument(
        "--min-buffer", type=int, default=3,
        help="最小帧间隔 (默认: 3)"
    )

    args = parser.parse_args()

    result = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        ffmpeg_exe=args.exe,
        ratio=args.ratio,
        min_buffer=args.min_buffer,
        resize_factor=args.resize_factor,
        output_format=args.output_format,
        keep_temp=args.keep_temp,
    )

    if not result.get("success"):
        exit(1)


if __name__ == "__main__":
    main()