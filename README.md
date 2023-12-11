# Self-supervised Learning for DNA sequences with circular dilated convolutional networks

paper: [https://doi.org/10.1016/j.neunet.2023.12.002](https://doi.org/10.1016/j.neunet.2023.12.002).

### Overview of cdilDNA with self-supervised learning.
![image](https://github.com/wiedersehne/cdilDNA/assets/8848011/17f25d7b-3370-4e85-97ab-1394e9bbe854)

## Variant Effect Prediction in Human Tissues. 
### To pretrain a model you need to follow the steps:
1. Download GRCH38 from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz. (3.1G)
2. Run **generate_pretrain_human.py** in `./data/`. Sequence length [1k, 5k, 10k] and numbers(100k) are required. You need to load two data files, **hg38.fa** and **chromosomes.csv**, for this task.
3. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in **config.yaml**.
### To fine-tune a pretrained model, you need to:
1. Run **generate_data.py** in `./data/` to save data. The sequence length is required. A total of 97,922 sequence will be extracted. **ve_df.csv** needs to be loaded.
2. Run **classify_lightning.py** to load the pretrained model under the folder `./human/100k_pretrain_best_epoch_9_8_16.pt` and train cdilDNA.
### Experimental Results

| Model                       | Parameters(k) | Running time (ms) | Memory (MB) | AUROC(%)  |
|-----------------------------|---------------|-------------------|-------------|-----------| 
| CDIL w/ pretrain            | 39.8          | 14.51             | 65.22       | **88.08** | 
| CDIL w/o pretrain           | 39.8          | 14.52             | 65.22       | 87.50     | 
| CNN                         | 39.8          | 11.60             | 49.03       | 85.42     | 
| Transformer                 | 33.3          | 66.65             | 190.05      | 66.49     |
| Nyströmformer w/ pretrain   | 42            | 35.08             | 133.19      | 87.71     |
| Nyströmformer w/o  pretrain | 42            | 35.08             | 133.19      | 87.50     |
| Performer                   | 41.3          | 24.25             | 76.44       | 87.32     |
| Enformer                    | /             | /                 | /           | 84.53     |

[varant_effect result](human_len.pdf)

## OCR Prediciton in Plants.

### To pretrain a model you need to follow the steps:
1. Download reference genome files from https://plantdeepsea-toturial2.readthedocs.io/en/latest/08-Statistics.html to `./data/plants/plant_genome/`.
2. Run **plant_download.ipynb** in `./data/plants/plant_data/` to download plant data.
3. Run **_cython_setup.sh** in `./plants/data_cython/`.
4. Run **run_pre.py** in `./plants/` for required plant.

### To fine-tune a pretrained model, you need to:
1. Run **run_epoch.py** in `./plants/` with **--pre_training**. The pretrained models are saved in `./plants/pre/`.

### Experimental Results
| Plant                      | A.thaliana | B.distachyon | O.sativa-MH | O.sativa-ZS | S.italica | S.bicolor | Z.mays    |
|----------------------------| ---------- | ------------ |-------------| ----------- |-----------| --------- |-----------|
| CDIL w/ pretrain           | **92.90**  | **93.47**    | **93.53**   | **93.09**   | **94.43** | **96.42** | **97.21** |
| CDIL w/o pretrain          | 90.39      | 91.95        | 91.45       | 91.66       | 92.42     | 95.80     | 95.87     |
| CNN                        | 82.02      | 87.09        | 86.05       | 84.94       | 88.30     | 91.52     | 89.79     |
| PlantDeepSEA               | 90.24      | 90.59        | 91.08       | 90.18       | 92.39     | 94.86     | 95.19     |
| Transformer                | 76.74      | 85.63        | 83.17       | 83.56       | 86.27     | 88.90     | 84.78     |
| Nyströmformer w/ pretrain  | 89.80      | 92.24        | 90.75       | 90.50       | 92.75     | 95.45     | 95.20     |
| Nyströmformer w/o pretrain | 86.02      | 90.19        | 86.92       | 86.31       | 90.94     | 93.67     | 89.77     |
| Performer                  | 75.95      | 85.25        | 82.81       | 83.02       | 86.60     | 88.09     | 85.13     |

[//]:![image](https://github.com/wiedersehne/cdilDNA/assets/8848011/304e2ce2-f009-4366-94c9-1f3427e530a7)
