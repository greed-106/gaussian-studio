#!/usr/bin/env python3
import subprocess
import shutil
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional, Callable, Union

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, log_file: Optional[Path] = None, silent: bool = False):
    """运行命令，失败则抛出异常。支持空格路径。"""
    if not silent:
        print(f">> {' '.join(map(str, cmd))}")
    
    # 如果指定了日志文件，重定向输出
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Command: {' '.join(map(str, cmd))}\n")
            f.write(f"{'='*80}\n\n")
            subprocess.run(cmd, cwd=cwd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, cwd=cwd, check=True)

def run_colmap_sfm(
    source: Union[str, Path],
    colmap_exe: Optional[Union[str, Path]] = None,
    no_gpu: bool = False,
    skip: bool = False,
    camera: str = "PINHOLE",
    log: Optional[Callable[[str], None]] = None,
    log_file: Optional[Union[str, Path]] = None
):
    """执行 SfM 流程。假设图像在 <source>/images 且无畸变。"""
    
    src = Path(source).resolve()
    imgs = src / "images"
    db = src / "database.db"
    sparse = src / "sparse"
    
    if not imgs.exists():
        raise FileNotFoundError(f"Images not found: {imgs}")

    # 查找可执行文件
    exe = Path(colmap_exe).resolve() if colmap_exe else Path(shutil.which("colmap") or "")
    if not exe.exists():
        raise FileNotFoundError("Colmap executable not found. Please specify --colmap_executable.")
    
    _log = log or print
    _log_file = Path(log_file) if log_file else None
    _log(f"Start SfM: {src}")

    if not skip:
        sparse.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature Extraction
        run_cmd([exe, "feature_extractor", "--database_path", str(db), "--image_path", str(imgs),
                 "--ImageReader.single_camera", "1", "--ImageReader.camera_model", camera,
                 "--SiftExtraction.use_gpu", "0" if no_gpu else "1"], cwd=src, log_file=_log_file, silent=True)
        
        # 2. Matching
        run_cmd([exe, "exhaustive_matcher", "--database_path", str(db),
                 "--SiftMatching.use_gpu", "0" if no_gpu else "1"], cwd=src, log_file=_log_file, silent=True)
        
        # 3. Mapper
        run_cmd([exe, "mapper", "--database_path", str(db), "--image_path", str(imgs),
                 "--output_path", str(sparse), "--Mapper.ba_global_function_tolerance", "0.000001"], cwd=src, log_file=_log_file, silent=True)
    
    # 4. Normalize Output (Ensure '0' exists)
    if not sparse.exists(): raise RuntimeError("No sparse output generated.")
    
    scenes = [d for d in sparse.iterdir() if d.is_dir()]
    if not scenes: raise RuntimeError("No scenes generated.")
    
    target = sparse / "0"
    if not target.exists():
        _log(f"Renaming {scenes[0].name} -> 0")
        scenes[0].rename(target)
    
    # Cleanup others
    for s in sparse.iterdir():
        if s.is_dir() and s.name != "0": shutil.rmtree(s)
    
    files = [f.name for f in target.iterdir() if f.is_file()]
    if not files: raise RuntimeError("Output directory is empty.")
    
    _log(f"Done! Output: {target}")
    return {"success": True, "path": target, "files": files}

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("--colmap_executable", "--exe", default="")
    p.add_argument("--no_gpu", action="store_true")
    p.add_argument("--skip_matching", action="store_true", dest="skip")
    p.add_argument("--camera", default="PINHOLE")
    args = p.parse_args()

    try:
        run_colmap_sfm(
            source=args.source_path,
            colmap_exe=args.colmap_executable or None,
            no_gpu=args.no_gpu,
            skip=args.skip,
            camera=args.camera
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        exit(1)