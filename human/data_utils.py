import numpy as np
from kipoiseq import Interval
from torch.utils.data import Dataset
from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer


class vcf_Dataset(Dataset):
    def __init__(self, ref, alt, tissue, label):
        self.ref, self.alt, self.tissue, self.label = ref, alt, tissue, label

    def __getitem__(self, index):
        ref = self.ref[index]
        alt = self.alt[index]
        tissue = self.tissue[index]
        label = self.label[index].float()
        return ref, alt, tissue, label

    def __len__(self):
        return len(self.label)
