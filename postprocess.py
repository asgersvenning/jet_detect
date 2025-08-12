import csv
import json
import os

from typing import Callable, Iterable

def combine_json_to_csv(
        input_dir : str, 
        output_csv: str, 
        prog : Callable[[Iterable[str]], Iterable[str]]
    ) -> None:
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f'Supplied input directory is not a valid directory: {input_dir}')
    if os.path.exists(output_csv):
        raise FileExistsError(f'Supplied output file already exists: {output_csv}')
    files = sorted([os.path.join(input_dir, rf) for rf in os.listdir(input_dir) if rf.lower().endswith('.json')])
    if not files:
        raise FileNotFoundError(f'No JSON files found in directory: {input_dir}')
    with open(files[0], 'r', encoding='utf-8') as fh:
        first_obj = json.load(fh)
    cols = list(first_obj.keys()) + ['origin']

    with open(output_csv, 'w', newline='', encoding='utf-8') as fh_out:
        w = csv.DictWriter(fh_out, fieldnames=cols)
        w.writeheader()
        for f in prog(files):
            with open(f, 'r', encoding='utf-8') as fh:
                obj = json.load(fh)
            row = {
                k : (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                for k, v in obj.items()
            }
            row['origin'] = os.path.basename(f)
            w.writerow(row)

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
        "-o", "--outpath", type=str, default=None, required=False,
        help=
        "Output path for combined CSV. "
        "A reasonable default is chosen if not passed. "
        "The file is overwritten if it already exists."
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=1,
        help="Verbosity, 0=silent, 1=minimal (default), 2=info, 3=debug"
    )

    return vars(parser.parse_args())

def main(year : int, result_dir : str, outpath : str | None=None, verbose : int=1):
    # Prepare environment
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

    if outpath is None:
        outpath = f'abisko_{year}.csv'
    if os.path.exists(outpath):
        os.remove(outpath)
    
    # Combine JSONs
    combine_json_to_csv(os.path.join(result_dir, f'abisko_{year}'), outpath, TQDM)

if __name__ == "__main__":
    main(**cli())
