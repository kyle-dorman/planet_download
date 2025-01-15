# Download Planet Data

Repo for downloading PlanetScope data based on a set of predefined grids. 

## Initial Setup

Some one time repo initialization work. 


### Clone repo
```bash
git clone git@github.com:kyledorman/planet_download.git
```

### Per environment variables
Copy the default env file and then edit the values inside. Importantly add your Planet API key.
```bash
cp base.env .env
```
Edit `.env` file.

### Install Miniconda

Follow directions [HERE](https://docs.anaconda.com/miniconda/install/)

For Mac, perfer to install miniconda using brew. 
```bash
brew doctor
brew update
brew upgrade
brew upgrade --cask --greedy
brew install --cask miniconda
```

### Install GDAL

Follow diriections [HERE](https://gdal.org/en/stable/download.html)

It is unclear if this is needed!

### Create conda environment
```bash
conda env create -f environment.yml
conda activate planet_download
```

## Download data

### Activate conda environment
```bash
conda activate planet_download
```

### Create a config file
See the file `src/config.py` to see all configuration options. You ***MUST*** set the `grid_dir` and `save_dir` variables. See `test_config.yaml` for a simple example.

### Run download script
Inpsect `run.sh` script to see how it can be used.
```bash
./run.sh
```

Download files for paritcular month/year combinations
```bash
./run.sh test_config.yaml 2020,2021,2022 09,10,11
```

Files will be saved to directories following the convention `SAVE_DIR/YEAR/MONTH/GRID_ID`. 

## Inspect Reuslts

Launch jupyter notebook
```bash
conda activate planet_download
jupyter notebook --notebook-dir=notebooks --port=8892
```
--allow-root --ip=0.0.0.0 --no-browser

Run the notebook `inspect.ipynb` to visualize the downloaded results. 

If you want to inspect the results for a particular grid you will need to extract the grid intermediates by running:
```bash
python src/inspect_grid_outputs.py --month MONTH --year YEAR --config-file CONFIG_FILE --grid-id GRID-ID
```

## Format code

Run
```bash
conda activate planet_download
./lint.sh
```

## Update dependencies
After changing the environment.yml file, run
```bash
conda activate planet_download
conda env update --file environment.yml --prune
```

## TODOs
- Test on the lab computer
- windows commands instead of linux
- sort grids everywhere
- add retry to udm dowloads
- find grid that is in 10 and 11 zone (channel island)
- more visibility into download process
- filter udm paths based in .tif everywhere
- log done in each script
- cleanup temp directory in selct_udms
- visibility into download progress
- change jupyter start to cmd
- start over for base in jupyter notebook and add explanation
- better descriptions of what file paths mean
- add explaination of file paths in test_config
- log exceptions on error or report number of errors
- normalize each channel separaetly 