import os
import re
import traceback
from argparse import ArgumentParser
from glob import glob

import torch
from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
from tqdm import tqdm as TQDM

from utils import parse_unknown_arguments


def config():
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

def predict(model, images : list[str], batch_size : int=1):
    from ultralytics import YOLO
    model = YOLO(model=model)
    model.to(device=torch.device("cuda:0"))
    model.eval()
    batch = []
    i = 0
    with torch.inference_mode():
        for lp, rp in TQDM(images, desc="Running batch inference...", unit="img"):
            batch.append(lp)
            if i == (len(images) - 1) or len(batch) == batch_size:
                for result in model.predict(batch, stream=True, verbose=False, conf=0.1):
                    yield result
                batch = []
            i += 1

if __name__ == "__main__":
    args = config()

    # input_images = search_input(args["input"])
    with IOHandler() as io:
        output_dir = args["output"]
        io.cd(args["input"])
        rpi = RemotePathIterator(io, clear_local=True, store=True, batch_parallel=10, n_local_files=1024, max_queued_batches=6)
        dst_check = {p : name for p in rpi.remote_paths if is_image(p) and not os.path.exists(os.path.join(output_dir, (name := os.path.splitext("__".join(p.split("/")))[0]) + ".txt"))}
        rpi.remote_paths = [p for p in rpi.remote_paths if p in dst_check]
        dst_files = [dst_check[p] for p in rpi.remote_paths]

        if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
            raise NotADirectoryError(
                f'Output directory ({output_dir}) is not a valid existing directory.'
                'Please make sure to create the output directory and that it is not a file.'
            )

        results = predict(args["weights"], rpi)
        from ultralytics.engine.results import Results
        for dst_name in dst_files:
            try:
                result = next(results)
                assert isinstance(result, Results)
                if args["visualize"]:
                    raise NotImplementedError("Disabled.")
                dst = os.path.join(output_dir, dst_name + ".txt")
                if os.path.exists(dst):
                    os.remove(dst)
                if len(result) == 0:
                    with open(dst, "w") as f:
                        f.writelines([""])
                else:
                    result.save_txt(txt_file=dst, save_conf=True)
            except StopIteration:
                break
            except Exception:
                with open(os.path.join(output_dir, dst_name + ".err"), "w") as f:
                    f.write(traceback.format_exc())
    