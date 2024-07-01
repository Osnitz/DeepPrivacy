import pathlib
import os
import typing
import argparse
import numpy as np
import torch
from deep_privacy import logger
from deep_privacy.inference.deep_privacy_anonymizer import DeepPrivacyAnonymizer
from deep_privacy.build import build_anonymizer, available_models
from face_detection.retinaface.detect import RetinaNetDetector
from face_detection import base, torch_utils
from face_detection.retinaface.config import cfg_mnet,cfg_re50
from face_detection.retinaface.models.retinaface import RetinaFace
video_suffix = [".mp4"]
image_suffix = [".jpg", ".jpeg", ".png"]


def recursive_find_file(folder: pathlib.Path,
                        suffixes: typing.List[str]
                        ) -> typing.List[pathlib.Path]:
    files = []
    for child in folder.iterdir():
        if not child.is_file():
            child_files = recursive_find_file(child, suffixes)
            files.extend(child_files)
        if child.suffix in suffixes:
            files.append(child)
    return files


def get_target_paths(source_paths: typing.List[pathlib.Path],
                     target_path: str,
                     default_dir: pathlib.Path):
    if not target_path is None:
        target_path = pathlib.Path(target_path)
        assert len(source_paths) == 1, \
            f"Target path is 1 file, but expected several inputs" + \
            f"target path={target_path}, source_path={source_paths}"
        target_path.parent.mkdir(exist_ok=True)
        return [target_path / p.name for p in source_paths]
    logger.info(
        f"Found no target path. Setting to default output path: {default_dir}")
    default_target_dir = default_dir
    target_path = default_target_dir
    target_path.mkdir(exist_ok=True, parents=True)
    target_paths = []
    for source_path in source_paths:
        if source_path.suffix in video_suffix:
            target_path = default_target_dir.joinpath("anonymized_videos")
        else:
            target_path = default_target_dir.joinpath("anonymized_images")
        target_path = target_path.joinpath(source_path.name)
        os.makedirs(target_path.parent, exist_ok=True)
        target_paths.append(target_path)
    return target_paths


def get_source_files(source_path: str):
    source_path = pathlib.Path(source_path)
    assert source_path.is_file() or source_path.is_dir(),\
        f"Did not find file or directory: {source_path}"
    if source_path.is_file():
        return [source_path]
    relevant_suffixes = image_suffix + video_suffix
    file_paths = recursive_find_file(source_path, relevant_suffixes)
    return file_paths


def init_anonymizer(cfg, generator):
    return DeepPrivacyAnonymizer(
        generator, **cfg.anonymizer, cfg=cfg)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_path", default=None,
        help="Path to the config. If not None, will override model_type"
    )
    parser.add_argument(
        "-m", "--model", default=available_models[0],
        choices=available_models,
        help="The anonymization model to be used."
    )
    parser.add_argument(
        "-s", "--source_path",
        help="Target path to infer. Can be video or image, or directory",
        default="test_examples/images"
    )
    parser.add_argument(
        "-t", "--target_path",
        help="Target path to save anonymized result.\
                Defaults to subdirectory of config file."
    )
    parser.add_argument(
        "--step", default=None, type=int,
        help="Set validation checkpoint to load. Defaults to most recent"
    )
    parser.add_argument(
        "--opts", default=None, type=str,
        help='can override default settings. For example:\n' +
        '\t opts="anonymizer.truncation_level=5, anonymizer.batch_size=32"')
    parser.add_argument(
        "--start_time", default=0, type=int,
        help="Start time for anonymization in case of video input. By default, the whole video is anonymized"
    )
    parser.add_argument(
        "--end_time", default=None, type=int,
        help="Start time for anonymization in case of video input. By default, the whole video is anonymized"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()    
    anonymizer, cfg = build_anonymizer(
        args.model, opts=args.opts, config_path=args.config_path,
        return_cfg=True)
    output_dir = cfg.output_dir
    source_paths = get_source_files(args.source_path)
    video_paths = [source_path for source_path in source_paths
                   if source_path.suffix in video_suffix]
    image_paths = [source_path for source_path in source_paths
                   if source_path.suffix in image_suffix]
    video_target_paths = []
    if len(video_paths) > 0:
        video_target_paths = get_target_paths(video_paths, args.target_path,
                                              output_dir)
    image_target_paths = []
    if len(image_paths) > 0:
        image_target_paths = get_target_paths(
            image_paths, args.target_path,
            output_dir)
    assert len(image_paths) == len(image_target_paths)
    assert len(video_target_paths) == len(video_paths)
    for video_path, video_target_path in zip(video_paths, video_target_paths):
        anonymizer.anonymize_video(video_path,
                                   video_target_path,
                                   start_time=args.start_time,
                                   end_time=args.end_time)
    if len(image_paths) > 0:
        anonymizer.anonymize_image_paths(image_paths, image_target_paths)

class face_detection(base.Detector):
    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_dir = "/home/matthieup/.cache/torch/hub/checkpoints/"
        map_location = torch_utils.get_device()

        if model == "mobilenet":
            cfg = cfg_mnet
            state_dict = self.load_local_or_remote_state_dict(
                "RetinaFace_mobilenet025.pth",
                model_dir=model_dir,
                map_location=map_location
            )
        else:
            assert model == "resnet50"
            cfg = cfg_re50
            state_dict = self.load_local_or_remote_state_dict(
                "RetinaFace_ResNet50.pth",
                model_dir=model_dir,
                map_location=map_location
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        net = RetinaFace(cfg=cfg)
        net.eval()
        net.load_state_dict(state_dict)
        self.cfg = cfg
        self.net = net.to(self.device)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.prior_box_cache = {}

    def load_local_or_remote_state_dict(self, filename, model_dir, map_location):
        local_path = os.path.join(model_dir, filename)
        if os.path.exists(local_path):
            state_dict = torch.load(local_path, map_location=map_location)
        else:
            raise FileNotFoundError(f"The model file {filename} does not exist in {model_dir}")
        return state_dict


if __name__ == "__main__":
    RetinaNetDetector.__init__= face_detection.__init__
    main()
