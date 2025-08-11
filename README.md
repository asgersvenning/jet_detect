# One-click setup for fresh environments
```sh
git clone https://github.com/asgersvenning/jet_detect.git
cd jet_detect
sudo bash setup_ucloud.sh
# Simply follow the instructions, then the inference will run by itself
```

# Setup and train
1. Unzip `data.zip` file in the root directory (e.g. `unzip data.zip -d .`).
2. Install `micromamba`/`mamba`/`conda` and run `[micromamba/mamba/conda] create -f conda.yaml`.
3. Run `[micromamba/mamba/conda] activate jet_detect`.
4. Run `python preprocess.py`.
5. Run `python train.py`.

# Inference
1. Follow steps in **Setup and train** (optionally omit step 4/5 if you already have a model).
2. Run `python predict.py -i <INPUT_DIR_FILE_OR_GLOB> -o <OUTPUT_DIR> -w <PATH_TO_WEIGHTS>.pt`.