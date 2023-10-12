import pandas as pd
import torch
import pyfaidx
import kipoiseq
import numpy
import numpy as np
import random
from kipoiseq import Interval
from torch.utils.data import Dataset
from Bio import SeqIO
from sklearn.preprocessing import LabelBinarizer


class FastaStringExtractor:
    """
        Extract a sequence based on the variant position and an interval
    """
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


fasta_file = './data/hg38.fa'
fasta_extractor = FastaStringExtractor(fasta_file)
# hg38_dict = SeqIO.to_dict(SeqIO.parse("./data/hg38.fa", "fasta"))


def generate_one(chr, pos, ref, alt, length, lb):
    """
        Extract one pair of reference and alternative sequence, and one-hot encode them.
    """
    variant = kipoiseq.Variant(chr, pos, ref, alt)
    interval = kipoiseq.Interval(variant.chrom, variant.start, variant.start).resize(length)
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
    center = interval.center() - interval.start
    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)
    reference_code = np.array(lb.transform(list(reference.lower())))
    alternate_code = np.array(lb.transform(list(alternate.lower())))
    return reference_code.T, alternate_code.T


def generat_data(data, length, split):
    """
        Build dataset which includes all the reference and alternative sequences, tissues and the variant labels.

    """
    ref_all = []
    alt_all = []
    tissue_all = []
    label_all = []
    gene = ['a','t','c','g','n']
    lb = LabelBinarizer() # DNA bases to one-hot encoding
    lb.fit(list(gene))
    for _, seq in data.iterrows():
        ref, alt = generate_one(seq['chrom'], seq['pos'], seq['reference'], seq['alternative'], length, lb)
        tissue = seq['tissue']
        label = seq['label']
        ref_all.append(ref)
        alt_all.append(alt)
        tissue_all.append(tissue)
        label_all.append(label)

    print(sum(label_all))

    ref_all = torch.from_numpy(numpy.array(ref_all)).to(torch.float16)
    alt_all = torch.from_numpy(numpy.array(alt_all)).to(torch.float16)
    tissue_all = torch.from_numpy(numpy.array(tissue_all)).to(torch.int8)
    label_all = torch.from_numpy(numpy.array(label_all)).to(torch.float16)
    torch.save(ref_all, f"./data/ref_{length}_{split}_noX.pt")
    torch.save(alt_all, f"./data/alt_{length}_{split}_noX.pt")
    torch.save(tissue_all, f"./data/tissue_{length}_{split}_noX.pt")
    torch.save(label_all, f"./data/label_{length}_{split}_noX.pt")
    return ref_all, alt_all, tissue_all, label_all

# generat_data(pd.read_csv('./data/ve_train.csv'), 1000, "train")
# generat_data(pd.read_csv('./data/ve_test.csv'), 1000, "test")

df = pd.read_csv('./data/ve_df.csv')

generat_data(pd.read_csv(f'./data/data/train_noX.csv'), 1000, "train")
generat_data(pd.read_csv(f'./data/data/test_noX.csv'), 1000, "test")
generat_data(pd.read_csv(f'./data/data/valid_noX.csv'), 1000, "valid")
