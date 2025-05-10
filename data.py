# from kipoiseq import Interval
from Bio import SeqIO
import kipoiseq
import numpy as np
# import time
import pyBigWig as pbw
from cooler import Cooler
from cooltools.lib.numutils import adaptive_coarsegrain

class SequenceData():
    def __init__(self, fasta_path):
        print(f'Loading fasta_seq: {fasta_path}')
        self.fasta_path = fasta_path
        self.seq_dict = self._load_fasta()

    def _load_fasta(self):
        seq_dict = {}
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            seq_dict[record.id] = record.seq
        return seq_dict

    def get(self, chr_name, start, end) -> np.float32:
        # chromosome_length = self._chromosome_sizes[chr_name]
        # if start < 0:
        #     print(f'start not acceptable: {chr_name}, {start}, {end}')  
        # elif end > chromosome_length:
        #     print(f'end not acceptable: {chr_name}, {start}, {end}')           
        sequence = str(self.seq_dict[chr_name][start:end].upper())
        onehot_seq = self.onehot_encode(sequence)
        # if len(onehot_seq) != 4_194_304:
        #     print(f'onehot_seq error: {len(sequence)}; chr_name:{chr_name}; start:{start}; end:{end}; begin base:{sequence[0:100]}')  
        return onehot_seq

    def onehot_encode(self, sequence):
        seq_emb = kipoiseq.transforms.functional.one_hot_dna(sequence, ('A', 'T', 'C', 'G', 'N'), '')
        # seq_emb = kipoiseq.transforms.functional.one_hot_dna(sequence, ('A', 'C', 'G', 'T', 'N'), 'N', 1)
        # seq_emb = kipoiseq.transforms.functional.one_hot_dna(sequence, ('A', 'C', 'G', 'T', 'N'),'')  ## np.float32 is default type!
        return seq_emb    



class TrackData():
    def __init__(self, epi_path, norm):
        # print(f'Loading Tracks: {epi_path}')
        self.path = epi_path
        self.norm = norm

    def get(self, chr_name, start, end):
        with pbw.open(self.path) as bw_file:
            signal = np.array(bw_file.values(chr_name, int(start), int(end)))
        signal = np.nan_to_num(signal, 0) # Important! replace nan with 0
        if self.norm == 'log':
            signal = np.log(signal + 1)
        elif self.norm is None:
            pass
        else:
            raise Exception(f'Norm type {self.norm} undefined')
        return signal

    def length(self, chr_name):
        with pbw.open(self.path) as bw_file:
            length = bw_file.chroms(chr_name)
        return length
    


class HiCData():
    def __init__(self, hic_path):
        print(f'Loading Hi-C: {hic_path}')
        self.hic = self.load_hic(hic_path)

    def get(self, chr_name, start, end):
        query = ((chr_name, start, end),)
        hic_mat =self.hic.matrix(balance='obj_weight').fetch(*query).astype(np.float32)
        hic_count =self.hic.matrix(balance=False).fetch(*query)
        mat = adaptive_coarsegrain(hic_mat, hic_count, max_levels=12)
        return mat

    def load_hic(self, path):
        return Cooler(path) 


class HiCData_withoutCG(HiCData):
    def __init__(self, hic_path):
        super(HiCData_withoutCG, self).__init__(hic_path)
        print(f'Loading Hi-C: {hic_path}')
        self.hic = self.load_hic(hic_path)

    def get(self, chr_name, start, end):
        query = ((chr_name, start, end),)
        hic_mat =self.hic.matrix(balance='obj_weight').fetch(*query).astype(np.float32)
        return hic_mat


