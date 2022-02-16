# Geoplus
## Requirements
- Python 3.x (3.8.8)
- Tensorflow 2.3.2
- Numpy 1.18.5
- scikit-learn 0.24.1
- Argparse 1.4.0
- prettytable 2.1.0 
## Installation
Please clone this repository as follows:
```
git clone https://github.com/daiyun02211/Geoplus.git
cd ./Geoplus
```
Please see also R package [**Geo2vec**](https://github.com/daiyun02211/Geo2vec) for feature extraction:
To install Geo2vec from Github, please use the following command in R consol.
```
if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")

devtools::install_github("daiyun02211/Geo2vec")
```
## Usage
# Single-nucletide m6A predictor (GepSe and i-GepSe)
An example raw data (genomic coordinates formated as .bed file) can be found in Example/base_predictor. Two-step preprocessing are required: 
1. step1_generate_encoding.R generates sequence encoding and Geo2vec encoding using R package [**Geo2vec**](https://github.com/daiyun02211/Geo2vec);
2. step2_rds2npy.py converts generated encodings to suitable Python format for modeling.
Python codes for GepSe and i-GepSe can be found in Scripts/Gepse and saved weights can be found in Weights/base_predictor:
```
python Scripts/GepSe/main.py --mode infer --data_dir ./Examples/base_predictor/processed/ --geo_enc chunkTX --tx long --cp_dir ./Weights/base_predictor/GepSe/
python Scripts/GepSe/main.py --mode infer --data_dir ./Examples/base_predictor/processed/ --geo_enc chunkTX --tx all --cp_dir ./Weights/base_predictor/iGepSe/
```
Optional arguments are provided to ease usage:
- ``--mode``: Three modes can be selected: train, eval and infer;
- ``--data_dir``: The directory where the processed data is stored;
- ``--geo_enc``: The Geo2vec encoding type should be consistent with the generated encoding in preprocessing;
- ``--tx``: The transcripts selection approach: None(only sequence), long(sequence + encoding of longest transcript), all(sequence + encoding of all mapped transcripts)
- ``--cp_dir``: The directory where the trained network weights (checkpoints) are stored.
Further arguments can be found:
```
python Scripts/GepSe/main.py -h
```
# Tissue-specific m6A predictor (ti-GepSe)
Please note that ti-GepSe was trained on MeRIP-seq data with instance length 50. Therefore, the default ti-GepSe provides prediction at up to 50-nt resolution.
Python codes for ti-GepSe can be found in Scripts/tiGepse and saved weights can be found in Weights/tissue_predictor:
```
python Scripts/tiGepSe/main.py --mode infer --data_dir ./Examples/tissue_predictor/ --tissue Lung --len 50 --cp_dir ./Weights/tissue_predictor/
```
Optional arguments are provided to ease usage:
- ``--mode``: Three modes can be selected: train, eval and infer;
- ``--data_dir``: The directory where the tissue data is stored. input_dir will be automaticlly generated with ``--data_dir`` and ``--tissue``;
- ``--tissue``: One of the 25 tissue types;
- ``--len``: The instance length. MeRIP-seq peak data is divided into instances for modeling using multiple instance learning; 
- ``--cp_dir``: The directory where the trained network weights (checkpoints) are stored. cp_path will be automaticlly generated with ``--cp_dir`` and ``--tissue``.
Further arguments can be found:
```
python Scripts/tiGepSe/main.py -h
```
