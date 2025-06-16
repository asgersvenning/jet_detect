import os
import re
from argparse import ArgumentParser
from collections.abc import Iterator
from glob import glob

from tqdm import tqdm as TQDM
from ultralytics import YOLO
from ultralytics.engine.results import Results

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

def predict(model : YOLO, images : list[str], batch_size : int=16):
    if len(images) < batch_size:
        return model(images)
    for i in TQDM(range(0, len(images), batch_size), desc="Running batch inference...", unit="batch"):
        batch_slice = slice(i, min(len(images), i + batch_size))
        for result in model(images[batch_slice], verbose=False):
            yield result

if __name__ == "__main__":
    args = config()
    
    model = YOLO(model=args["weights"])
    input_images = search_input(args["input"])
    output_dir = args["output"]

    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        raise NotADirectoryError(
            f'Output directory ({output_dir}) is not a valid existing directory.'
            'Please make sure to create the output directory and that it is not a file.'
        )

    results : Iterator[Results] = predict(model, input_images)
    for result, src in zip(results, input_images):
        if args["visualize"]:
            result.save(filename=os.path.join(output_dir, os.path.splitext(os.path.basename(src))[0] + ".jpg"))
        dst = os.path.join(output_dir, os.path.splitext(os.path.basename(src))[0] + ".txt")
        if os.path.exists(dst):
            os.remove(dst)
        result.save_txt(txt_file=dst, save_conf=True)