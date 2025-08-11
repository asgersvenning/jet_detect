import json
import os
import re
import traceback
from argparse import ArgumentParser
from collections import deque
from glob import glob
from typing import Any, Iterable, Iterator, Literal

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, Future
from io import BytesIO
import cv2, math

import numpy as np
import torch
from PIL import ExifTags, Image
from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
from tqdm import tqdm as TQDM

from utils import parse_unknown_arguments

SKIP_FILES = set([])

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

def predict(
    model: str,
    images: RemotePathIterator,
    *,
    batch_size: int = 64,
    imgsz: int = 1280,
    precision: "Literal['fp32','fp16','bf16']" = "bf16",
    device: str | int = "0",
    io_workers: int | None = None,
    prefetch_batches: int = 4,
) -> Iterator[Any]:
    from ultralytics import YOLO
    from ultralytics.data.loaders import LoadImagesAndVideos, SourceTypes

    if isinstance(device, int):
        device = str(device)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if prefetch_batches < 1:
        raise ValueError("prefetch_batches must be >= 1")
    if io_workers is None:
        io_workers = max(4, (os.cpu_count() or 8))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    class _PrefetchDataset(LoadImagesAndVideos):
        def __init__(
            self,
            it: Iterable[tuple[str, str]],
            meta_q: deque[dict[str, Any]],
            bs: int,
            workers: int,
            depth: int,
        ):
            self.it = it
            self._meta_q = meta_q
            self.mode = "image"
            self.bs = int(bs)
            self.count = 0
            self.source_type = SourceTypes(stream=False, screenshot=False, from_img=True, tensor=False)

            try:
                self._n = len(it)  # may fail for generators
            except Exception:
                self._n = None

            self._exec = ThreadPoolExecutor(max_workers=int(workers))
            self._buf: deque[Future] = deque()
            self._depth = int(depth) * self.bs
            self._iter = None
            self._exhausted = False

        def __len__(self):
            if self._n is None:
                return 0
            return (self._n + self.bs - 1) // self.bs

        def __iter__(self):
            self.count = 0
            self._exhausted = False
            self._iter = iter(TQDM(self.it, desc="Prefetching...", unit="img"))
            self._fill()
            return self

        def _submit(self, lp: str, rp: str):
            def _job():
                with open(lp, "rb") as f:
                    data = f.read()
                with Image.open(BytesIO(data)) as pim:
                    md = get_metadata(pim)
                md["FileName"] = rp
                arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR uint8
                return lp, arr, md
            self._buf.append(self._exec.submit(_job))

        def _fill(self):
            while len(self._buf) < self._depth and not self._exhausted:
                try:
                    lp, rp = next(self._iter)
                    self._submit(lp, rp)
                except StopIteration:
                    self._exhausted = True
                    break

        def __next__(self):
            if not self._buf and self._exhausted:
                self._exec.shutdown(wait=True, cancel_futures=False)
                raise StopIteration

            paths, im0s, s = [], [], []
            while len(paths) < self.bs:
                if not self._buf:
                    if self._exhausted:
                        break
                    self._fill()
                    if not self._buf and self._exhausted:
                        break

                lp, arr, md = self._buf.popleft().result()
                self._meta_q.append(md)
                paths.append(lp)
                im0s.append(arr)
                s.append("")
                self.count += 1

                # keep producer busy while we consume
                if len(self._buf) < self._depth // 2 and not self._exhausted:
                    self._fill()

            if not paths:
                self._exec.shutdown(wait=True, cancel_futures=False)
                raise StopIteration
            return paths, im0s, s

    yolo = YOLO(model=model)
    yolo.predictor = yolo._smart_load("predictor")(
        overrides={
            "conf": 0.1,
            "batch": int(batch_size),
            "imgsz": int(imgsz),
            "rect": False,
            "save": False,
            "verbose": False,
            "device": device,
            "half": precision == "fp16",
        },
        _callbacks=yolo.callbacks,
    )
    yolo.predictor.setup_model(model=yolo.model, verbose=False)

    amp = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision == "fp16"
        else nullcontext()
    )

    meta_q: deque[dict[str, Any]] = deque()
    loader = _PrefetchDataset(images, meta_q, bs=batch_size, workers=io_workers, depth=prefetch_batches)

    with torch.inference_mode(), amp:
        for r in yolo.predictor(loader, stream=True):
            md = meta_q.popleft()
            r.metadata = md
            yield r

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
        n_max : int | None=None
    ):
    with IOHandler() as io:
        io.cd(input)
        rpi = RemotePathIterator(io, clear_local=True, store=True, batch_parallel=14, n_local_files=4096, max_queued_batches=32)
        dst_check = {
            p : name for p in rpi.remote_paths 
            if is_image(p) 
            and not os.path.exists(os.path.join(output, (name := os.path.splitext("__".join(p.split("/")))[0]) + ".json"))
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
                dst = os.path.join(output, dst_name + ".json")
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
    
