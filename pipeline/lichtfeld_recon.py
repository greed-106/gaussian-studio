#!/usr/bin/env python3
"""
LichtFeld Studio Runner (Final Minimalist Version)
- Defaults for flags are False (disabled).
- No extra_args option.
- Direct list construction for clarity.
"""

import subprocess
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional, Union, Dict, Any

def run_lichtfeld_training(
    executable: Union[str, Path],
    data_path: Union[str, Path],
    output_path: Union[str, Path],
    iterations: Optional[int] = None,
    max_cap: Optional[int] = None,
    images_folder: Optional[str] = None,
    strategy: Optional[str] = None,
    tile_mode: Optional[int] = None,
    enable_mip: bool = False,          # Default: Disabled
    bilateral_grid: bool = False,      # Default: Disabled
    ppisp: bool = False,               # Default: Disabled
    headless: bool = False,            # Default: Disabled (GUI enabled)
    mask_mode: Optional[str] = None,
    max_width: Optional[int] = None
) -> Dict[str, Any]:
    """
    执行 LichtFeld Studio 训练。
    
    Args:
        executable: LichtFeld-Studio.exe 路径。
        data_path: 训练数据路径。
        output_path: 输出路径。
        iterations: 迭代次数 (--iter)。
        max_cap: MCMC 最大高斯数 (--max-cap)。
        images_folder: 图像文件夹名 (--images)。
        strategy: 优化策略 (--strategy)。
        tile_mode: 分块模式 (--tile-mode)。
        enable_mip: 启用 MIP 过滤 (默认 False)。
        bilateral_grid: 启用双边网格 (默认 False)。
        ppisp: 启用 PPISP (默认 False)。
        headless: 无头模式 (默认 False)。
        mask_mode: 遮罩模式 (--mask-mode)。
        max_width: 最大图像宽度 (--max-width)。
    """
    exe = Path(executable).resolve()
    data = Path(data_path).resolve()
    out = Path(output_path).resolve()

    if not exe.exists(): 
        raise FileNotFoundError(f"Executable not found: {exe}")
    if not data.exists(): 
        raise FileNotFoundError(f"Data path not found: {data}")
    
    out.mkdir(parents=True, exist_ok=True)

    # 1. 基础命令
    cmd = [str(exe), "--data-path", str(data), "--output-path", str(out)]

    # 2. 可选数值/字符串参数
    if iterations is not None:
        cmd.extend(["--iter", str(iterations)])
    if max_cap is not None:
        cmd.extend(["--max-cap", str(max_cap)])
    if images_folder:
        cmd.extend(["--images", images_folder])
    if strategy:
        cmd.extend(["--strategy", strategy])
    if tile_mode is not None:
        cmd.extend(["--tile-mode", str(tile_mode)])
    if mask_mode:
        cmd.extend(["--mask-mode", mask_mode])
    if max_width is not None:
        cmd.extend(["--max-width", str(max_width)])

    # 3. 布尔标志 (默认 False，仅当为 True 时添加)
    if enable_mip: 
        cmd.append("--enable-mip")
    if bilateral_grid: 
        cmd.append("--bilateral-grid")
    if ppisp: 
        cmd.append("--ppisp")
    if headless: 
        cmd.append("--headless")

    # 4. 执行
    print(f">> Running: {' '.join(cmd)}")
    try:
        # cwd 设为 exe 所在目录，确保相对路径资源加载正常
        subprocess.run(cmd, cwd=exe.parent, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Training failed with exit code {e.returncode}") from e
    except FileNotFoundError:
        raise FileNotFoundError(f"System cannot find the specified file: {exe}")

    return {"success": True, "output_path": out, "command": cmd}

# --- CLI Entry Point ---

if __name__ == "__main__":
    parser = ArgumentParser(description="LichtFeld Studio Wrapper")
    
    # 必填
    parser.add_argument("--exe", required=True, help="Path to LichtFeld-Studio.exe")
    parser.add_argument("-d", "--data-path", required=True, help="Training data path")
    parser.add_argument("-o", "--output-path", required=True, help="Output path")
    
    # 可选参数
    parser.add_argument("-i", "--iter", type=int, dest="iterations", help="Iterations")
    parser.add_argument("--max-cap", type=int, help="Max Gaussians")
    parser.add_argument("--images", type=str, help="Images folder name")
    parser.add_argument("--strategy", choices=["mcmc", "adc"], help="Strategy")
    parser.add_argument("--tile-mode", type=int, choices=[1, 2, 4], help="Tile mode")
    parser.add_argument("--mask-mode", type=str, help="Mask mode")
    parser.add_argument("--max-width", type=int, help="Max image width")
    
    # 布尔标志 (默认不启用，需手动添加参数才生效)
    parser.add_argument("--enable-mip", action="store_true", help="Enable MIP filter")
    parser.add_argument("--bilateral-grid", action="store_true", help="Enable bilateral grid")
    parser.add_argument("--ppisp", action="store_true", help="Enable PPISP")
    parser.add_argument("--headless", action="store_true", help="Headless mode")

    args = parser.parse_args()

    try:
        run_lichtfeld_training(
            executable=args.exe,
            data_path=args.data_path,
            output_path=args.output_path,
            iterations=args.iterations,
            max_cap=args.max_cap,
            images_folder=args.images,
            strategy=args.strategy,
            tile_mode=args.tile_mode,
            enable_mip=args.enable_mip,
            bilateral_grid=args.bilateral_grid,
            ppisp=args.ppisp,
            headless=args.headless,
            mask_mode=args.mask_mode,
            max_width=args.max_width
        )
        print("\n[SUCCESS] Training completed.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)