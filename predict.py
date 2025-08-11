import json
import os
import re
import traceback
from argparse import ArgumentParser
from collections import deque
from glob import glob
from typing import Any, Iterable, Iterator

import torch
from PIL import ExifTags, Image
from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
from tqdm import tqdm as TQDM
import numpy as np

from utils import parse_unknown_arguments

SKIP_FILES = set([
    "Dryas Cameras Nuolja 2020/Dryas ABIS-04/20200703/102_WSCT/WSCT9566.JPG",
    "Dryas Cameras Nuolja 2020/Dryas ABIS-04/20200703/102_WSCT/WSCT9567.JPG",
    "Dryas Cameras Nuolja 2020/Dryas ABIS-04/20200703/102_WSCT/WSCT9568.JPG",
    "Dryas Cameras Nuolja 2020/Dryas ABIS-08/20200624/101_WSCT/WSCT5690.JPG",
    "Dryas Cameras Nuolja 2020/Dryas ABIS-08/20200624/101_WSCT/WSCT5691.JPG",
    "Dryas Cameras Nuolja 2020/Dryas ABIS-08/20200624/101_WSCT/WSCT5692.JPG",
    "Dryas Cameras Nuolja 2020/Dryas ABIS-08/20200624/101_WSCT/WSCT5693.JPG"
])

def _proc_makernote(makernote : bytes):
    _, temperature, _, cameraID, _ = makernote.decode().split(":")
    temperature, cameraID = temperature.strip().replace("C", ""), cameraID.strip()
    return {"Temperature" : temperature, "CameraID" : cameraID}

def get_metadata(Image : Image) -> dict[str, Any]:
    exif = Image.getexif()
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    return {
        "DateTimeOriginal": exif_ifd.get(ExifTags.Base.DateTimeOriginal),
        "Flash": exif_ifd.get(ExifTags.Base.Flash),
        **_proc_makernote(exif_ifd.get(ExifTags.Base.MakerNote)),  # raw bytes
    }

IMAGE_PATTERN = re.compile(r'\.(jpe{0,1}g|png)$', re.IGNORECASE)
def is_image(file : str):
    return bool(re.search(IMAGE_PATTERN, file))

def search_input(src : str):
    if os.path.exists(src):
        if os.path.isfile(src):
            files = [src]
        else:
            files = [file for file in glob(os.path.join(src, "*"))]
    else:
        files = [file for file in glob(src)]
    files = list(filter(is_image, files))
    if len(files) == 0:
        raise RuntimeError(f'No images found in source: {src}')
    return files

def predict(model: str, images: Iterable[tuple[str, str]]) -> Iterator[Any]:
    # Ultralytics imports kept local to match your pattern
    from ultralytics import YOLO
    from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes

    class _IterLoader(LoadImagesAndVideos):
        """Minimal in-memory loader that passes Ultralytics isinstance() gate and streams 1 image/batch."""
        def __init__(self, it: Iterable[tuple[str, str]], out_q: deque):
            self.it = it
            self._q = out_q
            self.mode = "image"
            self.bs = 1
            self.count = 0
            # make check_source() happy when it reads .source_type for in_memory sources
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)

        def __iter__(self):
            self.count = 0
            # keep progress bar here so we don’t materialize anything
            self._it = iter(TQDM(self.it, desc="Running inference...", unit="img"))
            return self

        def __len__(self):
            return len(self.it)

        def __next__(self):
            lp, rp = next(self._it)  # local temp path, remote/original path
            im = Image.open(lp)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im0 = np.asarray(im)[:, :, ::-1]  # BGR for Ultralytics preprocessor
            # stash metadata payload for the outer coroutine
            self._q.append((rp, im))
            self.count += 1
            # paths must be a list[str]; s is a list[str] used for verbose printing
            return [lp], [im0], [""]

    yolo = YOLO(model=model)
    yolo.to(device=torch.device("cuda:0"))
    yolo.eval()
    yolo.predictor = yolo._smart_load("predictor")(
        overrides={
            "conf": 0.1,
            "batch": 1,
            "save": False,
            "mode": "predict",
            "rect": True,
            "verbose": False,
        },
        _callbacks=yolo.callbacks,
    )
    yolo.predictor.setup_model(model=yolo.model, verbose=False)

    q: deque[tuple[str, Image.Image]] = deque()
    loader = _IterLoader(images, q)

    with torch.inference_mode():
        for result in yolo.predictor(loader, stream=True):
            rp, im = q.popleft()  # remote/original path and the PIL image we opened
            metadata = get_metadata(im)
            metadata["FileName"] = rp
            result.metadata = metadata
            im.close()
            yield result

def save_result(result, dst : str, **kwargs):
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        if metadata is None:
            raise RuntimeError(f'No metadata found for results that would have been saved to "{dst}".')
        raise TypeError(f'Invalid metadata for results "{dst}", found: {metadata}')
    result_json = result.to_json(decimals=3)
    output_data = {
        **metadata,
        "results" : json.loads(result_json)
    }
    with open(dst, "w") as f:
        json.dump(output_data, f)

def cli():
    parser = ArgumentParser(prog = "train_yolov11", description="Predict with a YOLOv11 model", epilog="")
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to a directory with images, an image file or a glob to image files."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to result directory."
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True,
        help="Path to the weights of the model which you wish to perform inference with."
    )
    parser.add_argument(
        "-n", "--n_max", type=int, default=None, required=False,
        help="Maximum number of files to run inference for, default: all files in input. Useful for debugging."
    )
    parser.add_argument(
        "--visualize", action="store_true", required=False,
        help="Plot the predictions. Slow!"
    )
    args, extra = parser.parse_known_args()
    try:
        extra = parse_unknown_arguments(extra)
    except ValueError as e:
        raise ValueError(
                f"Error parsing extra arguments: `{' '.join(extra)}`. {e}\n\n"
                f"{parser.format_help()}"
            )
    return {**vars(args), **extra}

def main(
        input : str, 
        output : str, 
        weights : str, 
        n_max : int | None=None,
        visualize : bool=False
    ):
    with IOHandler() as io:
        io.cd(input)
        rpi = RemotePathIterator(io, clear_local=True, store=True, batch_parallel=10, n_local_files=512, max_queued_batches=6)
        dst_check = {
            p : name for p in rpi.remote_paths 
            if is_image(p) 
            and not os.path.exists(os.path.join(output, (name := os.path.splitext("__".join(p.split("/")))[0]) + ".txt"))
            and (p not in SKIP_FILES)
        }
        rpi.remote_paths = [p for p in rpi.remote_paths if p in dst_check]
        if n_max is not None and len(rpi.remote_paths) > n_max:
            rpi.remote_paths = rpi.remote_paths[:n_max]
        dst_files = [dst_check[p] for p in rpi.remote_paths]

        if not os.path.exists(output) or not os.path.isdir(output):
            raise NotADirectoryError(
                f'Output directory ({output}) is not a valid existing directory. '
                'Please make sure to create the output directory and that it is not a file.'
            )

        results = predict(weights, rpi)
        from ultralytics.engine.results import Results
        for dst_name in dst_files:
            try:
                result = next(results)
                assert isinstance(result, Results)
                if visualize:
                    raise NotImplementedError("Disabled.")
                dst = os.path.join(output, dst_name + ".txt")
                if os.path.exists(dst):
                    os.remove(dst)
                save_result(result, dst=dst, save_conf=True)
            except StopIteration:
                break
            except Exception:
                with open(os.path.join(output, dst_name + ".err"), "w") as f:
                    f.write(traceback.format_exc())

if __name__ == "__main__":
    main(**cli())
    