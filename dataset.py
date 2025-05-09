# from kipoiseq import Interval
import kipoiseq
import random
import pandas as pd
import numpy as np
from skimage.transform import resize 
from torch.utils.data import Dataset

from data import SequenceData, TrackData, HiCData, HiCData_withoutCG

# constants
RES_LENGTH = 1_000    # resolution of Hi-C data
INPUT_LENGTH = 4_194_304   ## 4 Mb
SCALE_LENGTH = 4_096  # scale 4_194 to 4_096 for Hi-C
STEP_LENGTH = 500_000   ## sampling step length for training
SHIFT_LENGTH = 500_000   ## romdomly shift [0~SHIFT_LENGTH] bp for sampling

RES_LENGTH2 = 16_000    # resolution of 32Mb model Hi-C data
INPUT_LENGTH2 = 33_554_432   
SCALE_LENGTH2 = 2_048  
STEP_LENGTH2 = 500_000   
SHIFT_LENGTH2 = 500_000  

# helper functions
def coin_flip():
    return random.random() > 0.5


class GenomeDataset(Dataset):
    def __init__(self, data_directory, input_tracklist, model, mode, fold='fold1', random_strand = True):
        self.data_directory = data_directory
        self.model = model
        if self.model is not None and self.model == 'SwinT4M':
            self.res = RES_LENGTH
            self.inputl = INPUT_LENGTH
            self.scalel = SCALE_LENGTH
            self.stepl = STEP_LENGTH
            self.shiftl = SHIFT_LENGTH
        else:
            self.res = RES_LENGTH2
            self.inputl = INPUT_LENGTH2
            self.scalel = SCALE_LENGTH2
            self.stepl = STEP_LENGTH2
            self.shiftl = SHIFT_LENGTH2        
        self.mode = mode
        self.fold = fold
        self.random_strand = random_strand
        self.chr_name_list = pd.read_csv(f'{self.data_directory}/../folds.txt', sep='\t')
        self.chr_names = self.chr_name_list[self.chr_name_list[self.fold] == self.mode]['chr'].unique().tolist()
        self.chr_lengths = self.chr_name_list[self.chr_name_list[self.fold] == self.mode]['length'].unique().tolist() 
        self.block_dict = self.load_block_regions(f'{data_directory}/../block.bed')
        self.seg_dict, self.cummulate_ids = self.get_data_dict()

        print("initializing rowsums")
        if self.model == 'SwinT4M':
           with open(f'{data_directory}/hic_matrix/rowsum1k.txt', 'r') as file:  
                self.rowsums = np.asarray([float(line.strip()) for line in file]) 
        else:
           with open(f'{data_directory}/hic_matrix/rowsum16k.txt', 'r') as file:  
                self.rowsums = np.asarray([float(line.strip()) for line in file]) 
                                   
        print(f'Loading multi-chromosome data ...') 
        self.seq_dict = SequenceData(f'{data_directory}/../dna_sequence/hg38.fa')
        self.input_tracklist = [TrackData(f'{data_directory}/genomic_features/{track_type}', norm) for (track_type, norm) in input_tracklist]
        self.mat_dict = self.get_mat_dict()
        print(f'multi-chromosome data loaded') 

    def __getitem__(self, id):
        chr_name, chr_id = self.get_chr_id(id)
        start, end = self.seg_dict[chr_name][chr_id]
        if self.mode == 'train': 
            start, end = self.single_shift(start, end)
        seq, input_tracks, mat1k = self.get_data_all(chr_name, start, end)
        # Apply noise only to rows where mode is 'train'
        if self.mode == 'train': 
            seq = self.gaussian_noise(seq, 0.1)
            input_tracks = [self.gaussian_noise(item, 0.1) for item in input_tracks]
        if self.random_strand:
            if coin_flip():
               seq, input_tracks, mat1k = self.reverse_comp(seq, input_tracks, mat1k)
        outputs = seq, input_tracks, mat1k # , mat1k, start, end, chr_name
        return outputs
        
    def __len__(self):
        return self.cummulate_ids[-1][1]
        
    def get_mat_dict(self):
        mat_dict = {}
        if self.model == 'SwinT4M':
            for chr_name in self.chr_names:
                mat_dict[chr_name] = HiCData(f'{self.data_directory}/hic_matrix/{chr_name}.mcool::/resolutions/{self.res}')
        else:
            for chr_name in self.chr_names:
                mat_dict[chr_name] = HiCData_withoutCG(f'{self.data_directory}/hic_matrix/{chr_name}.mcool::/resolutions/{self.res}')            
        return mat_dict

    def load_block_regions(self, path):
        df = pd.read_csv(path , sep = '\t', names = ['chr', 'start', 'end'])
        block_dict = {}
        for chr_name in self.chr_names:
            sub_df = df[df['chr'] == chr_name]
            regions = sub_df.drop('chr', axis = 1).to_numpy()
            block_dict[chr_name] = regions
        return block_dict

    def single_shift(self, start, end):
        offsets = random.randint(0, self.shiftl)
        start += offsets
        end += offsets
        return start, end
    
    def shift(self, segments):
        N = segments.shape[0]
        offsets = random.randint(0, self.shiftl, size=N)
        shift_segments = segments + offsets.reshape(-1, 1)
        return shift_segments

    def filter(self, segments, block_regions):
        filter_segments = []
        for start, end in segments:
            endmore = end + self.shiftl    
            start_cond = start <= block_regions[:, 1]
            end_cond = block_regions[:, 0] <= endmore
            if sum(start_cond * end_cond) == 0:
                filter_segments.append([start, end])
        return segments

    def get_data_dict(self):
        seg_dict = {}
        cummulate_ids = []
        start_id = 0
        for chr_name, chr_length in zip(self.chr_names, self.chr_lengths):
            segments = []
            block_regions = self.block_dict[chr_name]
            ## max_bin_num = (chr_length - INPUT_LENGTH - SHIFT_LENGTH) // STEP_LENGTH
            max_bin_num = (chr_length - self.inputl - self.shiftl) // self.stepl        
            starts = np.arange(0, max_bin_num).reshape(-1, 1) * self.stepl
            segments = np.append(starts, starts + self.inputl, axis=1)
            # segments = self.shift(segments)
            filter_segments = self.filter(segments, block_regions)
            seg_dict[chr_name] = filter_segments
            seg_num = len(filter_segments)
            cummulate_ids.append([start_id, start_id+seg_num])
            start_id += seg_num
        return seg_dict, cummulate_ids

    def get_chr_id(self, id):
        for i, chr_id in enumerate(self.cummulate_ids):
            start_id, end_id = chr_id
            if start_id <= id < end_id:
                return self.chr_names[i], id - start_id
            
    def gaussian_noise(self, inputs, std = 1):
        noise = np.random.randn(*inputs.shape) * std
        outputs = inputs + noise
        return outputs

    def reverse_comp(self, seq, input_tracks, mat1):
        seq_r = np.flip(seq, 0).copy() # n x 5 shape
        input_tracks_r = [np.flip(item, 0).copy() for item in input_tracks] # n
        mat1_r = np.flip(mat1, [0, 1]).copy()
        seq_r = self.complement(seq_r)  
        return seq_r, input_tracks_r, mat1_r
        
    def complement(self, seq):
        ## 'A', 'T', 'C', 'G', 'N'  -- >  'T', 'A', 'G', 'C', 'N'
        seq_comp = np.concatenate([seq[:, 1:2], seq[:, 0:1], seq[:, 3:4], seq[:, 2:3], seq[:, 4:5]], axis = 1)
        return seq_comp

    def get_data_all(self, chr_name, start, end):
        # Sequence processing
        # target_interval = kipoiseq.Interval(chr_name, start, end)
        # seq = self.seq.get(target_interval)
        seq = self.seq_dict.get(chr_name, start, end)
        # if len(seq) != INPUT_LENGTH:
        #      print(f'seq length:{len(seq)}; chr_name: {chr_name}; start: {start}; end: {end}')
        # Features processing
        input_tracks = [item.get(chr_name, start, end) for item in self.input_tracklist]
        # Hi-C matrix processing
        mat = self.mat_dict[chr_name].get(chr_name, start, end) 
        chr_id = int(chr_name[3:]) 
        mat /= self.rowsums[chr_id-1] 
        # self.check_length(seq, input_tracks[0], mat, chr_name, start, end)
        mat1k = resize(mat, (self.scalel, self.scalel), anti_aliasing=True)        ## 4096*4096
        return seq, input_tracks, mat1k



"""
import time
input_tracklist = {('ctcf_log2fc.bw', None), ('atac.bw','log')}
train_subset = GenomeDataset('/public/home/pku_yanglj/weijc/C.Origami-main/corigami_data/data/hg38/gm12878_1k_4195cleancools', input_tracklist, 'train', fold='fold1')
val_subset = GenomeDataset('/public/home/pku_yanglj/weijc/C.Origami-main/corigami_data/data/hg38/gm12878_1k_4195cleancools', input_tracklist, 'validate', fold='fold1') 
print(f'train_subset length is: {len(train_subset)}')
print(f'val_subset length is: {len(val_subset)}')
start_time = time.time()
seq, input_tracks, mat1k = train_subset[0]
print("access time:", time.time() - start_time)
print(f'my seq_round1 is: {seq.shape}')
print(seq)
seq, input_tracks, mat1k = train_subset[0]
print(f'my seq_round2 is: {seq.shape}')
print(seq)
print(f'my input_tracks is: {input_tracks[0].shape}')
print(input_tracks)
print(f'my mat1k is: {mat1k.shape}')
print(mat1k)
"""
