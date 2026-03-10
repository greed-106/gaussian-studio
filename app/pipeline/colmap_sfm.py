#!/usr/bin/env python3
import subprocess
import shutil
import struct
import collections
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Optional, Callable, Union, Dict, Tuple


# COLMAP data structures (from official script)
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}

CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file: Path) -> Dict:
    """
    Read COLMAP cameras.bin file (official implementation).
    
    Returns:
        Dictionary mapping camera_id to Camera namedtuple
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file: Path) -> Dict:
    """
    Read COLMAP images.bin file (official implementation).
    
    Returns:
        Dictionary mapping image_id to Image namedtuple
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix (official implementation)."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def extract_camera_parameters(sparse_dir: Path) -> Optional[Tuple[list, list]]:
    """
    Extract camera intrinsic and extrinsic parameters from COLMAP output.
    
    Args:
        sparse_dir: Path to COLMAP sparse reconstruction directory (e.g., sparse/0)
        
    Returns:
        Tuple of (intrinsic_matrix, extrinsic_matrix) as flattened lists, or None if failed
        - intrinsic_matrix: 3x3 matrix flattened to 9 elements [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        - extrinsic_matrix: 4x4 world-to-camera matrix flattened to 16 elements (row-major)
                           This is the view matrix from COLMAP, coordinate conversion handled in rendering
    """
    cameras_file = sparse_dir / "cameras.bin"
    images_file = sparse_dir / "images.bin"
    
    if not cameras_file.exists() or not images_file.exists():
        return None
    
    try:
        # Read cameras
        cameras = read_cameras_binary(cameras_file)
        if not cameras:
            return None
        
        # Get first camera
        camera = list(cameras.values())[0]
        params = camera.params
        
        # Build intrinsic matrix based on camera model
        if camera.model == "SIMPLE_PINHOLE":
            # params = [f, cx, cy]
            f, cx, cy = params
            fx = fy = f
        elif camera.model == "PINHOLE":
            # params = [fx, fy, cx, cy]
            fx, fy, cx, cy = params
        else:
            # For other models, try to extract focal length and principal point
            if len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            elif len(params) >= 3:
                f, cx, cy = params[:3]
                fx = fy = f
            else:
                return None
        
        # 3x3 intrinsic matrix flattened (row-major)
        intrinsic_matrix = [
            float(fx), 0.0, float(cx),
            0.0, float(fy), float(cy),
            0.0, 0.0, 1.0
        ]
        
        # Read images
        images = read_images_binary(images_file)
        if not images:
            return None
        
        # Get the image with the smallest image_id (usually the first frame)
        image = min(images.values(), key=lambda img: img.id)
        qvec = image.qvec
        tvec = image.tvec
        
        # Convert quaternion to rotation matrix
        R = qvec2rotmat(qvec)
        
        # Build 4x4 world-to-camera matrix [R | t; 0 0 0 1]
        # This is the view matrix from COLMAP
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = tvec
        
        # Flatten to list (row-major)
        extrinsic_matrix = extrinsic_matrix.flatten().tolist()
        
        return intrinsic_matrix, extrinsic_matrix
        
    except Exception as e:
        print(f"Error extracting camera parameters: {e}")
        return None

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
    """
    执行 SfM 流程，使用 COLMAP 自动重建器。
    
    Args:
        source: 工作目录路径，图像应在 <source>/images
        colmap_exe: COLMAP 可执行文件路径
        no_gpu: 是否禁用 GPU（当前未使用，自动重建器会自动检测）
        skip: 是否跳过重建（用于测试）
        camera: 相机模型（默认 PINHOLE）
        log: 日志回调函数
        log_file: 日志文件路径
        
    Returns:
        包含成功状态、输出路径和文件列表的字典
    """
    
    src = Path(source).resolve()
    imgs = src / "images"
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
        # 使用 COLMAP 自动重建器
        # 参数说明：
        # --workspace_path: 工作目录
        # --image_path: 图像目录
        # --quality: 重建质量 (low, medium, high, extreme)
        # --data_type: 数据类型 (video 表示视频帧序列)
        # --single_camera: 使用单相机模型
        # --sparse: 只进行稀疏重建
        # --dense: 不进行稠密重建
        # --camera_model: 相机模型
        run_cmd([
            exe, "automatic_reconstructor",
            "--workspace_path", str(src),
            "--image_path", str(imgs),
            "--quality", "high",
            "--data_type", "video",
            "--single_camera", "true",
            "--sparse", "true",
            "--dense", "false",
            "--camera_model", camera
        ], cwd=src, log_file=_log_file, silent=True)
    
    # 验证输出
    if not sparse.exists():
        raise RuntimeError("No sparse output generated.")
    
    scenes = [d for d in sparse.iterdir() if d.is_dir()]
    if not scenes:
        raise RuntimeError("No scenes generated.")
    
    # 确保输出目录名为 '0'
    target = sparse / "0"
    if not target.exists():
        _log(f"Renaming {scenes[0].name} -> 0")
        scenes[0].rename(target)
    
    # 清理其他场景目录
    for s in sparse.iterdir():
        if s.is_dir() and s.name != "0":
            shutil.rmtree(s)
    
    # 验证输出文件
    files = [f.name for f in target.iterdir() if f.is_file()]
    if not files:
        raise RuntimeError("Output directory is empty.")
    
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