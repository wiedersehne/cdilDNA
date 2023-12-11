# cdilDNA
![image](https://github.com/wiedersehne/cdilDNA/assets/8848011/17f25d7b-3370-4e85-97ab-1394e9bbe854)

## Variant Effect Prediction. 
### To pretrain a model you need to follow the steps:
1. Download GRCH38 from http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz. (3.1G)
2. Run **generate_pretrain_human.py** in `./data/`. Sequence length [1k, 5k, 10k] and numbers(100k) are required. You need to load two data files, **hg38.fa** and **chromosomes.csv**, for this task.
3. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in **config.yaml**.
### To fine-tune a pretrained model, you need to:
1. Run **generate_data.py** in `./data/` to save data. The sequence length is required. A total of 97,922 sequence will be extracted. **ve_df.csv** needs to be loaded.
2. Run **classify_lightning.py** to load the pretrained model under the folder `./human/100k_pretrain_best_epoch_9_8_16.pt` and train cdilDNA.
### Experimental Results

| Model | Parameters(k) | Running time (ms) | Memory (MB) | AUROC(%) |
| ------- | ----------- | -------- | ------- | ------------- | 
| CDIL w/pretrain    | 39.8       | 14.51    | 65.22   | **88.08**       | 
| CDIL w/o pretrain    | 39.8   |      14.52    |  65.22       | 87.50        | 
| CNN   | 39.8   |     11.60     |  49.03       | 85.42         | 
| Transformer   | 33.3   |    66.65      | 190.05      | 66.49         |
| Nystromformer w/pretrain   | 42   |    35.08      | 133.19      | 87.71       |
| Nystromformer w/o  pretrain   | 42   |    35.08      | 133.19     | 87.50        |
| Performer   | 41.3   |    24.25      | 76.44      | 87.32         |
| Enformer   |/   |    /      | /      | 84.53         |

[varant_effect result](human_len.pdf)

## OCRs prediciton in plants.
### To pretrain a model you need to follow the steps:
1. Download reference genome files from https://plantdeepsea-toturial2.readthedocs.io/en/latest/08-Statistics.html
2. Run **plant_download.ipynb** in `./data/plant_data/`to download training data.
3. Run **run.sh** in `./data/plant_generate/` to save data. The sequence length is required.
4. Run **pretraining.py** with the generated data. Configurations of different lengths shall be changed accordingly in **config_plant.yaml**.
### To fine-tune a pretrained model, you need to:
1. Run **run.sh** in `./data/plant_generate/` to generate data. The plant name is required.
2. Run **plant_classification.py** to load the pretrained model under the folder `./Pretrained_models/` and train cdilDNA.

### Experimental Results
| Plant               | A.thaliana | B.distachyon | O.sativa-MH | O.sativa-ZS | S.bicolor | S.italica | Z.mays |
| ------------------- | ---------- | ------------ | ----------- | ----------- | --------- | --------- | ------ |
| Number of OCR labels                | 19         | 9            | 15          | 15          | 14        | 9         | 19     |
| DeepSEA             | 92.02      | 92.88        | 92.95       | 92.19       | 96.24     | 94.04     | 96.64  |
| Nystromformer       | 89.22      | 90.86        | 89.08       | 88.10       | 94.50     | 91.61     | 90.74  |
| Linformer           | 70.56      | 83.50        | 79.28       | 80.43       | 87.30     | 84.64     | 80.82  |
| Transformer         | 64.96      | 82.53        | 78.79       | 78.62       | 85.15     | 84.24     | 63.02  |
| Mega                | 85.37      | 88.68        | 85.43       | 85.51       | 91.99     | 88.41     | 84.74  |
| S4                  | 85.82      | 90.70        | 88.30       | 87.84       | 93.95     | 90.84     | 92.87  |
| cdilDNA w/o         | 92.09      | 93.15        | 92.85       | 92.15       | 96.32     | 93.98     | 96.64  |
| cdilDNA w/ (1kbp)   | 92.24      | 93.57        | 93.42       | 92.81       | 96.41     | 94.33     | 97.07  |
| cdilDNA w/ (10kbp)  | 92.45      | 93.77        | 93.70       | 93.11       | 96.74     | 94.71     | 97.21  |
| cdilDNA w/ (50kbp)  | 92.81      | 93.79        | 93.83       | 93.28       | 96.68     | 94.79     | 97.31  |
| cdilDNA w/ (100kbp) | **93.22**  | **94.10**    | **93.99**   | **93.56**   | **96.88** | **95.08** | **97.32** |

