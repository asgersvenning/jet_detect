import csv
import os
import re
from collections import ChainMap, Counter
from datetime import datetime

from pyremotedata.implicit_mount import IOHandler


def count_lines(path : str):
    with open(path, 'r') as f:
        return sum(1 for _ in f)
    
def get_metadata(path : str):
    if "__" not in path:
        raise ValueError(f'Invalid path, no occurrences of "__" found in: {path}')
    
    try:
        parts = os.path.basename(path).split("__")
        remote_path = "/".join(parts)
        order, camera = parts[1:3]
        order = 1 + int(order.endswith("II"))
    except Exception as e:
        e.add_note(f'Failed to parse metadata from path: {path}')
        raise e
        
    return {
        "remote_path" : remote_path,
        "order" : order,
        "camera" : camera
    }

TIMESTAMP_PATTERN = re.compile(fr'^(\d\d:\d\d:\d\d \d\d-\d\d-\d\d\d\d)\s*')

def parse_timestamp(line : str):
    match = re.search(TIMESTAMP_PATTERN, line)
    if match is None:
        raise RuntimeError(f'Unable to parse timestamp from line: {line}')
    timestamp = match.groups()[0]
    return datetime.strptime(timestamp, "%H:%M:%S %d-%m-%Y")

def get_timestamps(io : IOHandler, path : str=".", pattern=r'\.JPG$'):
    pattern = re.compile(pattern)
    paths = io.execute_command(f'cls -1 --date --time-style="%H:%M:%S %d-%m-%Y" "{path}"')
    timestamps = dict()
    for p in paths:
        if not re.search(pattern, p):
            continue
        timestamp = parse_timestamp(p)
        timestamps[p.removeprefix(timestamp.strftime("%H:%M:%S %d-%m-%Y "))] = timestamp
    return timestamps

def cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="jet_postprocess",
        description="Postprocess and collect metadata for predictions from the image detection pipeline."
    )
    parser.add_argument(
        "-Y", "--year", type=int, required=True,
        help="Which year to postprocess, e.g. 2019."
    )
    parser.add_argument(
        "-r", "--result_dir", type=str, default="runs/erda",
        help=
        "Result directory containing the output subdirectories for each year (e.g. abisko_2019)."
        'Default: "runs/erda"'
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1,
        help="Verbosity, 0=silent, 1=minimal (default), 2=info, 3=debug"
    )

    return vars(parser.parse_args())

def main(year : int, result_dir : str, verbose : int=1):
    # Stage 0: Prepare environment
    if not (isinstance(year, int) and year > 1900 and year < 3000):
        raise ValueError(f'Invalid year: {year}')
    if not (isinstance(result_dir, str) and os.path.exists(result_dir) and os.path.isdir(result_dir)):
        raise ValueError(f'Invalid result directory: {result_dir}')
    if not (isinstance(verbose, int) and verbose >= 0):
        raise ValueError(f'Invalid verbosity: {verbose}')
    
    match verbose:
        case 0:
            def TQDM(iter, *args, **kwargs):
                return iter
        case n if n > 0:
            from tqdm import tqdm as TQDM
        case _:
            raise ValueError(f"unsupported verbosity: {verbose!r}")
    if verbose > 2:
        print("Environment prepared successfully.")

    # Stage 1: Parse result files
    root = os.path.join(result_dir, f"abisko_{year}")
    files = sorted([os.path.join(root, file) for file in os.listdir(root)])

    ext = Counter(os.path.splitext(file)[-1] for file in files)
    if verbose > 1:
        print(f"Found files: {ext}")

    files = [file for file in files if file.endswith(".txt")]
    lines = {file : count_lines(file) for file in TQDM(files, desc="Counting lines in files...", unit="file")}
    if verbose > 1:
        print(
            'Found number of files with linenumbers (linenumber: #files):\n\t"' +
            "\n\t".join(f'{e}: {c}' for e, c in sorted(Counter(lines.values()).items()))
        )

    metadata = {file : get_metadata(file) for file in TQDM(files, desc="Parsing metadata from filenames...", unit="file")}
    data = {col : [] for col in ["file", "remote_path", "order", "camera", "count"]}
    for file, meta in metadata.items():
        data["file"].append(file)
        data["count"].append(lines[file])
        for k, v in meta.items():
            data[k].append(v)

    dirs = sorted(list(set(map(lambda x : os.path.dirname(x.split(os.sep)[-1].replace("__", os.sep)), data["file"]))))

    if verbose > 2:
        print("Result files parsed successfully.")

    # Stage 2: Get timestamps from remote files
    all_timestamps = dict()

    with IOHandler(verbose=verbose>2) as io:
        io.cd("Timelapse/storage/tundraplants")
        io.cd(f"{year}/Abisko")
        io.time_stamp_pattern = re.compile("kasgji124123hfhjsngnahsgh8")
        all_timestamps = ChainMap(*(get_timestamps(io, d) for d in TQDM(dirs, "Resolving timestamps...", unit="dir")))

    data["timestamp"] = [
        all_timestamps[
            f.removeprefix(f"{result_dir}/abisko_{year}/").replace("__", "/").replace(".txt", ".JPG")
        ] 
        for f in data["file"]
    ]
    cols = ["file", "remote_path", "order", "camera", "count", "timestamp"]
    assert set(cols) == set(list(data.keys())), f'Inconsistent columns found in data: {list(data.keys())}'

    n_rows = set(map(len, data.values()))
    assert len(n_rows) == 1, f'Found inconsistent number of rows in data: {n_rows}'
    n_rows = list(n_rows)[0]

    if verbose > 2:
        print("Timestamps successfully retrieved.")

    # Stage 3: Save/Write data to CSV
    csv_path = f"test_result_{year}.csv"
    if verbose > 1:
        print(f'Writing data to: {csv_path}')
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(TQDM(zip(*[data[c] for c in cols]), total=n_rows, desc="Writing data to CSV...", unit="line"))
    
    if verbose > 0:
        print(f'Results saved to: {csv_path}')
    if verbose > 2:
        print("Results successfully saved.")

if __name__ == "__main__":
    main(**cli())
