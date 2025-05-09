import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from swint import CirculateSwinBlock as swin_1d_block

class SwinT4M(nn.Module):
    
    def __init__(self, input_track_num, mid_hidden = 128):
        super(SwinT4M, self).__init__()
        print('Initializing Model')
        self.encoder = Encoder(input_track_num, output_size = mid_hidden, num_blocks = 9)
        self.swin1d = swin1d_block(mid_hidden)
        self.decoder16k = Decoder(mid_hidden * 2 * 2)
        self.decoder8k = Decoder2(mid_hidden * 2 * 2 + 1)
        self.decoder4k = Decoder2(mid_hidden * 2 * 2 + 1)
        self.decoder2k = Decoder2(mid_hidden * 2 + 1)
        self.decoder1k = Decoder2(mid_hidden * 2 + 1) 

    def forward(self, x, r8, r4, r2, r1):
        x = self.transposef(x).float()   ## b,l,c  --> b,c,l
        x_1k = self.encoder(x)
        x_1k = self.transposef(x_1k)

        x_1k, x_2k, x_4k, x_8k, x_16k = self.swin1d(x_1k)
        start = r8 * 2
        x_8knew = x_8k[:, start:start + 256, :]
        start = start * 2 + r4 * 2
        x_4knew = x_4k[:, start:start + 256, :]
        start = start * 2 + r2 * 2
        x_2knew = x_2k[:, start:start + 256, :]
        start = start * 2 + r1 * 2
        x_1knew = x_1k[:, start:start + 256, :]

        x_1knew = self.transposef(x_1knew)
        x_2knew = self.transposef(x_2knew)
        x_4knew = self.transposef(x_4knew)
        x_8knew = self.transposef(x_8knew)
        x_16k = self.transposef(x_16k)
        x_1knew = self.diagonalize(x_1knew, 256)         ## 经过diagonalize操作第二维上的feature数翻倍
        x_2knew = self.diagonalize(x_2knew, 256)
        x_4knew = self.diagonalize(x_4knew, 256)   
        x_8knew = self.diagonalize(x_8knew, 256)
        x_16k = self.diagonalize(x_16k, 256)

        x_16k =  self.decoder16k(x_16k)
        x_8knew  =  self.decoder8k(x_8knew,  x_16k[:, :, r8:r8 + 128, r8:r8 + 128].detach())
        x_4knew  =  self.decoder4k(x_4knew,  x_8knew[:, :, r4:r4 + 128, r4:r4 + 128].detach())
        x_2knew  =  self.decoder2k(x_2knew,  x_4knew[:, :, r2:r2 + 128, r2:r2 + 128].detach())
        x_1knew  =  self.decoder1k(x_1knew,  x_2knew[:, :, r1:r1 + 128, r1:r1 + 128].detach())

        x_16k =  (0.5 * x_16k + 0.5 * x_16k.transpose(2, 3)).squeeze(1)
        x_8knew =  (0.5 * x_8knew + 0.5 * x_8knew.transpose(2, 3)).squeeze(1)
        x_4knew =  (0.5 * x_4knew + 0.5 * x_4knew.transpose(2, 3)).squeeze(1)
        x_2knew =  (0.5 * x_2knew + 0.5 * x_2knew.transpose(2, 3)).squeeze(1)
        x_1knew =  (0.5 * x_1knew + 0.5 * x_1knew.transpose(2, 3)).squeeze(1)       
        return x_16k, x_8knew, x_4knew, x_2knew, x_1knew

    def transposef(self, x):
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x, size):
        x_i = x.unsqueeze(2).repeat(1, 1, size, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, size)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map            



class SwinT32M(nn.Module):
    def __init__(self, input_track_num, mid_hidden = 128, use_checkpoint=True):
        super(SwinT32M, self).__init__()
        print('Initializing Model')
        self.encoder = Encoder(input_track_num, output_size = mid_hidden, num_blocks = 9)
        self.swin1d = swin1d_block(mid_hidden)
        self.newswin1d = swin1d_block2(mid_hidden * 2)

        self.decoder128k = Decoder(mid_hidden * 2 * 2)
        self.decoder64k = Decoder2(mid_hidden * 2 * 2 + 1)
        self.decoder32k = Decoder2(mid_hidden * 2 * 2 + 1)
        self.newdecoder16k = Decoder2(mid_hidden * 2 * 2 + 1)

        self.use_checkpoint = use_checkpoint
        self.newparams = set()

    def forward(self, x, r64, r32, r16):
        x = self.transposef(x).float()
        if self.use_checkpoint:
            # x = checkpoint.checkpoint(run0, x, dummy)
            x = checkpoint.checkpoint(self.encoder, x)
        else:   
            # x = run0(x, dummy)
            x = self.encoder(x)   
        x = self.transposef(x)

        x_1k, x_2k, x_4k, x_8k, x_16k = self.swin1d(x)
        x_32k, x_64k, x_128k = self.newswin1d(x_16k)
        start = r64 * 2
        x_64knew = x_64k[:, start:start + 256, :]
        start = start * 2 + r32 * 2
        x_32knew = x_32k[:, start:start + 256, :]
        start = start * 2 + r16 * 2
        x_16knew = x_16k[:, start:start + 256, :]

        x_16knew = self.transposef(x_16knew)
        x_32knew = self.transposef(x_32knew)
        x_64knew = self.transposef(x_64knew)
        x_128k = self.transposef(x_128k)

        x_16knew = self.diagonalize(x_16knew, 256)         ## 
        x_32knew = self.diagonalize(x_32knew, 256)
        x_64knew = self.diagonalize(x_64knew, 256)   
        x_128k = self.diagonalize(x_128k, 256)

        x_128k =  self.decoder128k(x_128k)
        x_64knew  =  self.decoder64k(x_64knew,  x_128k[:, :, r64:r64 + 128, r64:r64 + 128].detach())
        x_32knew  =  self.decoder32k(x_32knew,  x_64knew[:, :, r32:r32 + 128, r32:r32 + 128].detach())
        x_16knew  =  self.newdecoder16k(x_16knew,  x_32knew[:, :, r16:r16 + 128, r16:r16 + 128].detach())

        x_128k =  (0.5 * x_128k + 0.5 * x_128k.transpose(2, 3)).squeeze(1)
        x_64knew =  (0.5 * x_64knew + 0.5 * x_64knew.transpose(2, 3)).squeeze(1)
        x_32knew =  (0.5 * x_32knew + 0.5 * x_32knew.transpose(2, 3)).squeeze(1)
        x_16knew =  (0.5 * x_16knew + 0.5 * x_16knew.transpose(2, 3)).squeeze(1)      

        return x_128k, x_64knew, x_32knew, x_16knew 
            
    def replace_with_predicts(self, preckpt_path):
        # Load the state_dict of previous model
        prestate_dict = torch.load(preckpt_path)['state_dict']
        
        # Copy parameters from Model2 to Model1 and set requires_grad=False
        replaced_params = set()
        for name, param in prestate_dict.items():
            # Remove the 'model.' prefix from the parameter name  
            stripped_name = name[6:] if name.startswith('model.') else name  
            if stripped_name.startswith('encoder') or stripped_name.startswith('swin1d'):
                if stripped_name in self.state_dict():
                    current_param = self.state_dict()[stripped_name]
                    current_param.copy_(param)
                    current_param.requires_grad = False
                    replaced_params.add(stripped_name)
                    # print(f'replacing {stripped_name}')
        
        # Collect parameters that were not replaced
        for name, param in self.named_parameters():
            if name not in replaced_params:
                self.newparams.add(param)
        print ('Num. of total params.', len(list(self.named_parameters())))
        print ('Num. of prestated params.', len(replaced_params))
        print ('Num. of new   params.', len(self.newparams))

    def set_mode(self):
        # Set the modules with Model2 parameters to eval mode and others to train mode
        for name, module in self.named_modules():
            if any(name.startswith(prefix) for prefix in ['encoder', 'swin1d']):
                module.eval()
            else:
                module.train()

    def transposef(self, x):
        return x.transpose(1, 2).contiguous()

    def diagonalize(self, x, size):
        x_i = x.unsqueeze(2).repeat(1, 1, size, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, size)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map      



class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out


class Encoder(nn.Module):
    # def __init__(self, num_epi, output_size = 256, filter_size = 5, num_blocks = 12):
    def __init__(self, num_epi, output_size = 128, filter_size = 5, num_blocks = 9):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_epi = nn.Sequential(
                                    nn.Conv1d(num_epi, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )

        feature_list = [16, 16, 16, 16, 16, 32, 32, 64, 64, 64]
        self.res_blocks_seq = self.get_res_blocks(num_blocks, feature_list[:-1], feature_list[1:])
        self.res_blocks_epi = self.get_res_blocks(num_blocks, feature_list[:-1], feature_list[1:])
        self.conv_end = nn.Conv1d(128, output_size, 1)

    def forward(self, x):
        seq = x[:, :5, :]
        epi = x[:, 5:, :]
        seq = self.res_blocks_seq(self.conv_start_seq(seq))
        epi = self.res_blocks_epi(self.conv_start_epi(epi))
        x = torch.cat([seq, epi], dim = 1)
        out = self.conv_end(x)      ## JW: seq_length== 2*64
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks    
    

class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil 
        self.res = nn.Sequential(
                        nn.Dropout(0.1),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len, dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    ## nn.Dropout(p=0.02),
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks


class Decoder2(Decoder):   ## decoder that incorporates preditions from lower resolution decoder
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder2, self).__init__(in_channel)
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    ## nn.Dropout(p=0.02),
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='nearest')

    def forward(self, x, y):
        x = torch.cat([x, self.upsample(y)], axis=1)
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out
    

def swin1d_block(dim):

    window_size = 32
    stages = [
        (4, False, False, window_size),
        (2, True,  True,  window_size),
        (2, True, False, window_size),
        (2, True,  True,  window_size),
        (2, True,  True,  window_size),
    ]
    model = swin_1d_block(stages, dim)
    return model

def swin1d_block2(dim):

    window_size = 32
    stages = [
        (2, True,  True,  window_size),
        (2, True,  True,  window_size),
        (2, True,  True,  window_size),    
    ]
    model = swin_1d_block(stages, dim)
    return model
        
if __name__ == '__main__':
    main()

