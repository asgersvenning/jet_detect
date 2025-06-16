import csv
import os
from glob import glob
from collections import defaultdict
from ultralytics.utils.plotting import Annotator
from PIL import Image

from tqdm import tqdm as TQDM

def read_data(label_src : str):
    src_data = defaultdict(list)
    with open(label_src, "r") as f:
        reader = csv.reader(f, delimiter=",")
        columns = next(reader)
        if columns != COLUMNS:
            raise RuntimeError(f'Unexpected columns {columns} found. Expected {COLUMNS}.')
        for row in reader:
            for value, col in zip(row, columns):
                src_data[col].append(int(value) if col in COLUMNS[-2:] else float(value))
    return dict(src_data)

def write_yolo_labels(label_data : dict[str, int | float], dst : str):
    if os.path.exists(dst):
        if not OVERWRITE:
            raise RuntimeError(f'Reformatted file {dst} from {src} already exists.')
        os.remove(dst)
    with open(dst, "w") as f:
        writer = csv.writer(f, delimiter='\t')
        for i, (xc, yc, bw, bh) in enumerate(zip(*[label_data[col] for col in COLUMNS[:4]])):
            writer.writerow([0, xc, yc, bw, bh])

def plot_labels(label_data : dict[str, int | float], image_path : str, dst="test.jpg"):
    pil_im = Image.open(image_path)
    iw, ih = pil_im.size
    annot = Annotator(pil_im, line_width=10)
    for i, (xc, yc, bw, bh) in enumerate(zip(*[label_data[col] for col in COLUMNS[:4]])):
        xc *= iw
        bw *= iw
        yc *= ih
        bh *= ih
        x1, y1 = (xc - bw / 2), (yc - bh / 2)
        x2, y2 = x1 + bw, y1 + bh
        annot.box_label([x1, y1, x2, y2], str(i))
    annot.save(dst)

if __name__ == "__main__":
    COLUMNS = ["x_coord", "y_coord", "width", "height", "Row", "id"]
    OVERWRITE = True
    splits = glob(os.path.join("data", "*"))
    for split in splits:
        dirs = {os.path.basename(dir) : dir for dir in glob(os.path.join(split, "*"))}
        label_dir = dirs["labels"]
        image_dir = dirs["images"]
        for src in TQDM(glob(os.path.join(label_dir, "*.csv")), desc=f"Reformatting {split}..."):
            src_name = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(label_dir, f'{src_name}.txt')
            src_data = read_data(src)
            # plot_labels(src_data, os.path.join(image_dir, f'{src_name}.JPG'))
            write_yolo_labels(src_data, dst)