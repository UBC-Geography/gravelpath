# Lighttable analysis code

## Requirements

### Python

This code is written for Python 3.12 and packages managed with miniconda. The conda environment can be created with the following command:

```bash
conda env create -f environment.yml -n lighttable
```

### GPU support

The code should run fine on CPU and take advantage of multiprocessing. But for GPU support, the Nvidia CUDA toolkit is required. The code has been tested with CUDA 12.3.2 and cuDNN 8.9.7 (not required yet, but needed when doing machine learning).
The code uses OpenCV for image processing. To use a GPU with OpenCV functions, OpenCV might have to be compiled from source with the correct flags to allow CUDA.

## Usage

### Configuration

A sample configuration file is given with `example.toml`. For each run that is to be analyzed, a `.toml` file has to be created and saved in the `configs_to_run` directory.

The configuration file specifies the locations of input and output files, as well as the parameters for the analysis.

### Running the code

All configurations in the `configs_to_run` directory can be run with the following command, if the conda environment is activated:

```bash
conda activate lighttable
python run_all_configs.py
```

## Implementation roadmap

- [x] Read configuration files
- [x] Loop to load images with OpenCV
- [x] Filter image data and substract background image
- [x] Store found particles in a sqlite database for analysis later
- [ ] Analyze the particle data in the sqlite database
- [ ] Calculate particle size distribution
- [ ] Calculate sediment transport rate
