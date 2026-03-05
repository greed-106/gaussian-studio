#!/usr/bin/env python3
"""
Splat Compress Tool

将 PLY 格式的高斯点云转换为 SOG 格式。
使用 npx splat-transform 进行转换，支持失败重试。
"""

import subprocess
import platform
import shutil
import time
from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import Union, Dict, Any, Optional


def get_npx_command() -> str:
    """
    获取平台适配的 npx 命令。

    Returns:
        Windows 返回 'npx.cmd'，其他平台返回 'npx'
    """
    system = platform.system().lower()
    return "npx.cmd" if system == "windows" else "npx"


def compress_splat(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    max_retries: int = 3,
    log_file: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    执行 splat-transform 转换，将 PLY 转换为 SOG 格式。

    Args:
        input_path: 输入的 PLY 文件路径
        output_path: 输出的 SOG 文件路径
        max_retries: 最大重试次数（默认 3）
        log_file: 日志文件路径

    Returns:
        包含转换结果的字典
    """
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    # 检查输入文件
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # 确定输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 获取平台适配的 npx 命令
    npx = get_npx_command()

    # 检查 npx 是否可用
    npx_path = shutil.which(npx)
    if not npx_path:
        raise FileNotFoundError(
            f"npx not found. Please ensure Node.js is installed and npx is in PATH."
        )

    # 构建命令
    cmd = [npx, "splat-transform", str(input_file), str(output_file)]

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            # 如果指定了日志文件，重定向输出
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Command (attempt {attempt}): {' '.join(cmd)}\n")
                    f.write(f"{'='*80}\n\n")
                    subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
            else:
                subprocess.run(cmd, check=True)

            # 验证输出文件
            if output_file.exists():
                # print(f"[SUCCESS] Output file: {output_file}")
                return {
                    "success": True,
                    "input_path": str(input_file),
                    "output_path": str(output_file),
                    "output_size": output_file.stat().st_size,
                    "attempts": attempt,
                }
            else:
                raise RuntimeError(f"Output file was not created: {output_file}")

        except subprocess.CalledProcessError as e:
            last_error = RuntimeError(
                f"splat-transform failed with exit code {e.returncode}"
            )
            print(f"[WARN] Attempt {attempt} failed: exit code {e.returncode}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"System cannot find the specified command: {npx}"
            ) from e
        except Exception as e:
            last_error = e
            print(f"[WARN] Attempt {attempt} failed: {e}")

        # 如果不是最后一次尝试，等待一段时间后重试
        if attempt < max_retries:
            print(f"Retrying in 2 seconds...")
            time.sleep(2)

    # 所有重试都失败
    raise last_error or RuntimeError("Transform failed after all retries")


def main():
    """CLI 入口点"""
    parser = ArgumentParser(
        description="Splat Compress Tool - 将 PLY 转换为 SOG 格式",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  %(prog)s input.ply output.sog

  # 指定重试次数
  %(prog)s input.ply output.sog --retries 5

  # 使用完整路径
  %(prog)s "C:\\path\\to\\input.ply" "C:\\path\\to\\output.sog"
        """,
    )

    parser.add_argument("input", help="输入 PLY 文件路径")
    parser.add_argument("output", help="输出 SOG 文件路径")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        dest="max_retries",
        help="最大重试次数（默认: 3）",
    )

    args = parser.parse_args()

    try:
        result = compress_splat(
            input_path=args.input,
            output_path=args.output,
            max_retries=args.max_retries,
        )
        print(f"\n[SUCCESS] Transform completed.")
        print(f"Output: {result['output_path']}")
        print(f"Size: {result['output_size']:,} bytes")
        if result['attempts'] > 1:
            print(f"Attempts: {result['attempts']}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()
