import os
import cv2
import json
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from pycolmap import SceneManager
from tqdm import tqdm
from typing_extensions import assert_never

from ..normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


@dataclass
class ParserConfig:
    """Config for `Parser`, structured so it can be embedded in an OmegaConf tree."""

    data_dir: str
    factor: int = 1
    normalize: bool = False
    test_every: int = 8
    gaussian_ckpt_path: Optional[str] = None
    """Path to a simple_trainer.py checkpoint (.pth) to load reference Gaussian
    parameters (means, scales, quats, opacities, sh0, shN) from."""


class Parser:
    """COLMAP parser.

    Each `_init_*` method fills in one group of attributes and is called, in
    order, from `__init__`; later methods depend on attributes set by earlier
    ones (see the call order below).
    """

    def __init__(self, cfg: ParserConfig):
        self.cfg = cfg

        # Extended metadata used by the Bilarf dataset; not derived from COLMAP.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        # Bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])

        # Set by _init_cameras_and_extrinsics:
        self.image_names: List[str] = []                      # (num_images,)
        self.camtoworlds: np.ndarray = None                   # (num_images, 4, 4)
        self.camera_ids: List[int] = []                       # (num_images,)
        self.Ks_dict: Dict[int, np.ndarray] = {}              # camera_id -> K
        self.params_dict: Dict[int, np.ndarray] = {}          # camera_id -> distortion params
        self.imsize_dict: Dict[int, Tuple[int, int]] = {}     # camera_id -> (width, height)
        self.mask_dict: Dict[int, Optional[np.ndarray]] = {}  # camera_id -> mask
        self.camtype: str = "perspective"                     # "perspective" | "fisheye"

        # Set by _init_image_paths:
        self.image_paths: List[str] = []             # (num_images,)

        # Set by _init_points3d:
        self.points: np.ndarray = None                  # (num_points, 3)
        self.points_err: np.ndarray = None              # (num_points,)
        self.points_rgb: np.ndarray = None              # (num_points, 3)
        self.point_indices: Dict[str, np.ndarray] = {}  # image_name -> [M,]

        # Set by _init_normalization:
        self.transform: np.ndarray = np.eye(4)          # (4, 4)

        # Set by _init_undistortion_maps:
        self.mapx_dict: Dict[int, np.ndarray] = {}
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, list] = {}

        # Set by _init_scene_scale:
        self.scene_scale: float = 0.0

        # Set by _init_gaussians (only if cfg.gaussian_ckpt_path is set):
        self.gaussian_step: Optional[int] = None
        self.gaussian_means: Optional[np.ndarray] = None      # (num_gaussians, 3)
        self.gaussian_scales: Optional[np.ndarray] = None     # (num_gaussians, 3)
        self.gaussian_quats: Optional[np.ndarray] = None      # (num_gaussians, 4)
        self.gaussian_opacities: Optional[np.ndarray] = None  # (num_gaussians,)
        self.gaussian_sh0: Optional[np.ndarray] = None        # (num_gaussians, 1, 3)
        self.gaussian_shN: Optional[np.ndarray] = None        # (num_gaussians, K-1, 3)

        manager = self._init_scene_manager()

        self._init_cameras_and_extrinsics(manager)
        self._init_image_paths(manager)
        self._init_points3d(manager)
        self._init_normalization()
        self._rescale_intrinsics_to_actual_image_size()
        self._init_undistortion_maps()
        self._init_scene_scale()
        self._init_gaussians()

    def _init_scene_manager(self) -> SceneManager:
        colmap_dir = os.path.join(self.cfg.data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(self.cfg.data_dir, "sparse")
        assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()
        return manager

    def _init_cameras_and_extrinsics(self, manager: SceneManager) -> None:
        """Extract extrinsics (world-to-camera) and per-camera intrinsics/distortion."""
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            camera_id = im.camera_id
            camera_ids.append(camera_id)

            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= self.cfg.factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // self.cfg.factor, cam.height // self.cfg.factor)
            mask_dict[camera_id] = None
        print(f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.")

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        self.image_names = [image_names[i] for i in inds]
        self.camtoworlds = camtoworlds[inds]
        self.camera_ids = [camera_ids[i] for i in inds]
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        
        # NB: `camtype` reflects the *last* camera seen above, matching the
        # upstream behavior this was refactored from (a single distortion
        # model is assumed for undistortion regardless of per-camera type).
        self.camtype = camtype

    def _init_image_paths(self, manager: SceneManager) -> None:
        if self.cfg.factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{self.cfg.factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(self.cfg.data_dir, "images")
        image_dir = os.path.join(self.cfg.data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if self.cfg.factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=self.cfg.factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        self.image_paths = [
            os.path.join(image_dir, colmap_to_image[f]) for f in self.image_names
        ]

    def _init_points3d(self, manager: SceneManager) -> None:
        """3D points and {image_name -> [point_idx]}."""
        self.points = manager.points3D.astype(np.float32)
        self.points_err = manager.point3D_errors.astype(np.float32)
        self.points_rgb = manager.point3D_colors.astype(np.uint8)

        point_indices = dict()
        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        self.point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

    def _init_normalization(self) -> None:
        if not self.cfg.normalize:
            self.transform = np.eye(4)
            return

        camtoworlds = self.camtoworlds
        points = self.points

        T1 = similarity_from_cameras(camtoworlds)
        camtoworlds = transform_cameras(T1, camtoworlds)
        points = transform_points(T1, points)

        T2 = align_principal_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)

        transform = T2 @ T1

        # Fix for up side down. We assume more points towards the bottom of
        # the scene, which is true when the ground floor is visible.
        if np.median(points[:, 2]) > np.mean(points[:, 2]):
            # rotate 180 degrees around x axis such that z is flipped
            T3 = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            camtoworlds = transform_cameras(T3, camtoworlds)
            points = transform_points(T3, points)
            transform = T3 @ transform

        self.camtoworlds = camtoworlds
        self.points = points
        self.transform = transform

    def _rescale_intrinsics_to_actual_image_size(self) -> None:
        """Load one image to check the size. In the case of the tanksandtemples
        dataset, the intrinsics stored in COLMAP correspond to 2x upsampled images."""
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

    def _init_undistortion_maps(self) -> None:
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:  # no distortion
                continue
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert camera_id in self.params_dict, f"Missing params for camera {camera_id}"

            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if self.camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif self.camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(self.camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

    def _init_scene_scale(self) -> None:
        """Size of the scene, measured by the spread of the camera locations."""
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def _init_gaussians(self) -> None:
        """Load reference Gaussian parameters from a simple_trainer.py checkpoint,
        e.g. `<result_dir>/ckpts/ckpt_<step>_rank0.pt`. No-op if not configured."""
        if self.cfg.gaussian_ckpt_path is None:
            return

        ckpt = torch.load(self.cfg.gaussian_ckpt_path, map_location="cpu", weights_only=True)
        splats = ckpt["splats"]
        self.gaussian_step = ckpt["step"]
        self.gaussian_means = splats["means"].numpy()
        self.gaussian_scales = splats["scales"].numpy()
        self.gaussian_quats = splats["quats"].numpy()
        self.gaussian_opacities = splats["opacities"].numpy()
        self.gaussian_sh0 = splats["sh0"].numpy()
        self.gaussian_shN = splats["shN"].numpy()


@dataclass
class DatasetConfig:
    """Config for `Dataset`, structured so it can be embedded in an OmegaConf tree."""

    split: str = "train"
    patch_size: Optional[int] = None
    load_depths: bool = False


class Dataset:
    """A simple dataset class."""

    def __init__(self, parser: Parser, cfg: DatasetConfig):
        self.parser = parser
        self.cfg = cfg
        indices = np.arange(len(self.parser.image_names))
        if cfg.split == "train":
            self.indices = indices[indices % self.parser.cfg.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.cfg.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        # Images are distorted. Undistort them.
        if len(params) > 0:
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

            # Random crop
        if self.cfg.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.cfg.patch_size, 1))
            y = np.random.randint(0, max(h - self.cfg.patch_size, 1))
            image = image[y : y + self.cfg.patch_size, x : x + self.cfg.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K":          torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image":      torch.from_numpy(image).float(),
            "image_id":   item,  # the index of the image in the dataset
        }

        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.cfg.load_depths:
            # projected points to image plane to get depths
            worldtocams   = np.linalg.inv(camtoworlds)
            image_name    = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world  = self.parser.points[point_indices]
            points_cam    = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj   = (K @ points_cam.T).T
            points        = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths        = points_cam[:, 2]  # (M,)

            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data
