# Download Planet Data

Repo for downloading PlanetScope data based on a set of predefined grids. 

## Initial Setup

Some one time repo initialization work. 

### Install Conda
You can install miniconda if on Mac or Linux. On Windows install Anaconda.

#### Miniconda
Follow directions [HERE](https://docs.anaconda.com/miniconda/install/)

For Mac, perfer to install miniconda using brew. 
```bash
brew doctor
brew update
brew upgrade
brew upgrade --cask --greedy
brew install --cask miniconda
```

#### Anaconda
Follow directions [HERE](https://docs.anaconda.com/anaconda/install/)

### Open Terminal
On Windows, open an Andaconda Terminal, on Linux/Mac open a regular terminal. 

### Install Git
Check if git is installed. It will return something e.g. `/usr/bin/git` if git is installed. 
```bash
# Linux/Mac
which git
# Windows
where git
```

If git is not installed, install it. 
```bash
# Windows
conda install git
# Mac
brew install git
```

### Clone repo
This command will create a new folder `planet_download` in your terminal's current directory. If you want it installed somewhere specific, move to that folder first (`cd SOMEWHERE/ELSE`)
```bash
git clone git@github.com:kyledorman/planet_download.git
```

After cloning the repo, enter the folder
```bash
cd planet_download
```

### Planet API Key
The Planet API key is loaded from a file `.env`. 

Start by copying the default file
```bash
# Linux/Mac
cp base.env .env

# Windows
copy base.env .env
```
Look up your API key from the [Planet Account Settings](https://www.planet.com/account/#/user-settings). 

Edit `.env` file using a text editor and add your API key.

### Create conda environment
```bash
conda env create -f environment.yml
```

## Download data

### Activate conda environment
```bash
conda activate planet_download
```

### Create a config file
See the file `src/config.py` to see all configuration options. You ***MUST*** set the `grid_dir` and `save_dir` variables. See `config.yaml` for a simple example. Feel free to edit the `config.yaml` directly. 

`grid_dir` - The path to a folder of geojson grid files in the wgs84 CRS. 
`save_dir` - The path to a folder where you want to save the data. During the download process, data will be saved to folders following the convention: `save_dir/YEAR/MONTH/GRID_ID`

### Run download script
Inpsect `run.py` script to see how it can be used.
```bash
python src/run.py --help
```

Download files for paritcular month/year combinations
```bash
python src/run.py --config-file config.yaml --year 2020 --year 2021 --year 2022 --month 09 --month 10 --month 11
```
The above example would download data for 3 years (2020, 2021, 2022) and 3 months (Set, Oct, Nov) for a total of 9 year/month combindations. You must provide at least 1 year and 1 month. 

## Inspect Reuslts
You can inspect the results of a download using an included jupyter notebook. 

Launch jupyter notebook
```bash
conda activate planet_download
jupyter notebook --notebook-dir=notebooks --port=8892
```
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

### Install GDAL

Follow diriections [HERE](https://gdal.org/en/stable/download.html)

It is unclear if this is needed!

## TODOs
- Test on the lab computer
- How to approprietly handle grids in both 10 and 11 zone (25059125, 25059125)
