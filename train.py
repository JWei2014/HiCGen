
import sys
import torch
import torch.nn.functional as F 
import argparse
import numpy as np
# import pandas as pd
import random
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from scipy.interpolate import interp1d

import model as my_models
from dataset import GenomeDataset

def main():
    args = init_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("initializing distnorm")
    if args.pred_mode is not None and args.pred_mode == 'SwinT4M':
        distnorm_list = np.load(f'{args.dataset_root}/{args.dataset_celltype}/hic_matrix/new.1Kb.cooltools.res1000.npy') [:4195]  
        # backgd_bydist = np.exp(resize_1d(backgd_bydist, 4096, kind='cubic'))
        distnorm_list = resize_1d(distnorm_list, 4096, kind='cubic')
        backgd = distnorm_list[np.abs(np.arange(4096)[:, None] - np.arange(4096)[None, :])]
        backgd1k = torch.tensor(np.reshape(backgd[:256, :256], (256, 1, 256, 1)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device) 
        backgd2k = torch.tensor(np.reshape(backgd[:512, :512], (256, 2, 256, 2)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device) 
        backgd4k = torch.tensor(np.reshape(backgd[:1024, :1024], (256, 4, 256, 4)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device)  
        backgd8k = torch.tensor(np.reshape(backgd[:2048, :2048], (256, 8, 256, 8)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device)  
        backgd16k = torch.tensor(np.reshape(backgd[:4096, :4096], (256, 16, 256, 16)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device)  
        print("initializing distnorm done!") 
        training4M(args, backgd1k, backgd2k, backgd4k, backgd8k, backgd16k)
    elif args.pred_mode == 'SwinT32M':
        distnorm_list = np.load(f'{args.dataset_root}/{args.dataset_celltype}/hic_matrix/new.16Kb.cooltools.res16000.npy') [:2097]  
        distnorm_list = resize_1d(distnorm_list, 2048, kind='cubic')
        backgd = distnorm_list[np.abs(np.arange(2048)[:, None] - np.arange(2048)[None, :])]
        backgd16k = torch.tensor(np.reshape(backgd[:256, :256], (256, 1, 256, 1)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device) 
        backgd32k = torch.tensor(np.reshape(backgd[:512, :512], (256, 2, 256, 2)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device) 
        backgd64k = torch.tensor(np.reshape(backgd[:1024, :1024], (256, 4, 256, 4)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device)  
        backgd128k = torch.tensor(np.reshape(backgd[:2048, :2048], (256, 8, 256, 8)).mean(axis=1).mean(axis=2), dtype=torch.float32).unsqueeze(0).to(device)  
        print("initializing distnorm done!") 
        training32M(args, backgd16k, backgd32k, backgd64k, backgd128k)       

def resize_1d(data, new_size, kind='linear'):
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, new_size)
    interpolator = interp1d(x_old, data, kind=kind)
    resized_data = interpolator(x_new)
    return resized_data

def init_parser():
    parser = argparse.ArgumentParser(description='Training parser.')
 
    parser.add_argument('--save_path', dest='save_path', default='checkpoints', help='Path to the model checkpoint')
    parser.add_argument('--data-root', dest='dataset_root', default='../../corigami_data/data/hg38', help='Root path')
    parser.add_argument('--celltype', dest='dataset_celltype', default='gm12878_1k_4195withoutICE', help='cell type for prediction')
    parser.add_argument('--fold', dest='fold', default='fold1', help='fold ID')  
    parser.add_argument('--pred-mode', dest='pred_mode', default='SwinT4M', help='choose between SwinT4M, SwinT32M, Hyena4M, Hyena32M')

    parser.add_argument('--patience', dest='patience', default=60, type=int, help='Epoches before early stopping')
    parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=100, type=int, help='Max epochs')
    parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=1, type=int, help='Number of GPUs to use')
 
    parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--num-workers', dest='dataloader_num_workers', default=8, type=int, help='Dataloader workers')
    parser.add_argument('--ckpt_path', dest='ckpt_path', default=None, type=str, help='checkpoint filepath')
    parser.add_argument('--preckpt-path', dest='preckpt_path', default=None, type=str, help='checkpoint filepath for 16k res. model')
    parser.add_argument('--accumul-batch', dest='accumulate_grad_batches', default=4, type=int, help='accumulate number of batches') 
 
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    return args


def training4M(args, backgd1k, backgd2k, backgd4k, backgd8k, backgd16k):

    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=False, mode="min")
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.save_path}/models', save_top_k=20, monitor='val_loss')
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    csv_logger = pl.loggers.CSVLogger(save_dir = f'{args.save_path}/csv')
    all_loggers = csv_logger
    
    seed = random.randint(1, 5000)
    pl.seed_everything(seed, workers=True)
    pl_module = TrainModule4M(args, backgd1k, backgd2k, backgd4k, backgd8k, backgd16k)
    # if( args.ckpt_path is not None):
    #     print('loading:  ',   f'{args.ckpt_path}')
    #     pl_module.load_from_checkpoint(f'{args.ckpt_path}')
    pl_trainer = pl.Trainer(strategy='ddp',
                            accelerator="gpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            logger = all_loggers,
                            callbacks = [early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs  
                            # resume_from_checkpoint=args.resume_from_checkpoint  
                            )
    trainloader, validateloader, testloader = pl_module.get_dataloader(args)
    pl_trainer.fit(pl_module, trainloader, validateloader)


def training32M(args, backgd16k, backgd32k, backgd64k, backgd128k):

    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=False, mode="min")
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.save_path}/models', save_top_k=20, monitor='val_loss')
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')
    loggers = pl.loggers.CSVLogger(save_dir = f'{args.save_path}/csv')
    
    seed = random.randint(1, 5000)
    pl.seed_everything(seed, workers=True)
    pl_module = TrainModule32M(args, backgd16k, backgd32k, backgd64k, backgd128k)
    if( args.ckpt_path is not None):
        print('loading:  ',   f'{args.ckpt_path}')
        pl_module.load_from_checkpoint(f'{args.ckpt_path}')
    pl_trainer = pl.Trainer(strategy='dp',   ## strategy='ddp',
                            accelerator="gpu", devices=args.trainer_num_gpu,
                            gradient_clip_val=1,
                            accumulate_grad_batches=args.accumulate_grad_batches,
                            logger = loggers,
                            callbacks = [early_stop_callback,
                                         checkpoint_callback,
                                         lr_monitor],
                            max_epochs = args.trainer_max_epochs  
                            # resume_from_checkpoint=args.resume_from_checkpoint  #  
                            )
    trainloader, validateloader, testloader = pl_module.get_dataloader(args)
    pl_trainer.fit(pl_module, trainloader, validateloader)


class TrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

    def forward(self):
        raise Exception('Not implemented')

    def proc_batch(self, batch):         
        seq, input_tracks, mat = batch             ## , mat, start, end, chr_name
        # print(f"seq.shape: {seq.shape}")
        # print(f"pred_track.shape: {pred_track.shape}")
        tracks = torch.cat([track.unsqueeze(2) for track in input_tracks], dim = 2)
        inputs = torch.cat([seq, tracks], dim = 2)
        mat = mat.float()
        return inputs, mat

    def downsample_tensor(self, matrix):  

        #  * 2 for better long-range interactions 
        downsampled_tensor = F.avg_pool2d(matrix, kernel_size=2, stride=2) # * 2
        return downsampled_tensor 

    def downsample_mask(self, mask, level, threshold=0.9):  

        downsampled_mask = F.avg_pool1d(mask, kernel_size=level, stride=level)  

        downsampled_mask = (downsampled_mask >= threshold) # .float() 
        return downsampled_mask
      
    def generate_mask(self, matrix, threshold=0.001):  

        # non_zero_ratios = (matrix != 0).float().mean(dim=(1, 2))  
        non_zero_ratios =  (matrix != 0).float().mean(dim=1)
        mask = (non_zero_ratios >= threshold) # .float()  
        return mask  
    
    def sample_tensor(self, mat, mask, start, target_size=256, min_ratio=0.3):  
        iter0 = 0
        while True:  
            # indices = torch.randperm(mat32.size(1))[:16]  
            # r = torch.randint(0, 128, (1,)).item() ## [1,128)
            r = random.randint(0, 128)  ## [1,128]
            offset = start * 2 + r * 2 
            sampled_mat = mat[:, offset:offset+target_size, offset:offset+target_size]              
            sampled_mask = mask[:, offset:offset+target_size]  
            non_masked_ratio = sampled_mask.float().mean()  
            if non_masked_ratio >= min_ratio or iter0 > 20:  
                return sampled_mat, sampled_mask, r, offset 
            iter0 += 1 

    def check_if_invalid(self, mask, min_ratio=0.5):  
        if mask.float().mean() < min_ratio:  
            return True
        else:
            return False
             
    def diagonalize_mask(self, mask, nondiagonal, size=256):
        x_i = mask.unsqueeze(1).repeat(1, size, 1)
        x_j = mask.unsqueeze(2).repeat(1, 1, size)
        masknew = x_i * x_j * nondiagonal
        return masknew

    def diagonalize_mask2(self, mask, size=256):
        x_i = mask.unsqueeze(1).repeat(1, size, 1)
        x_j = mask.unsqueeze(2).repeat(1, 1, size)
        masknew = x_i * x_j
        return masknew
        
    def training_step(self, batch, batch_idx):  
        loss = self._shared_step(batch, batch_idx)  
        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, prog_bar=True)  ## batch_size = inputs.shape[0], 
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_step(batch, batch_idx)
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_step(batch, batch_idx)
        return ret_metrics
    
    def _shared_step(self, batch, batch_idx):
        raise Exception('Not implemented')

    # Collect epoch statistics
    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'train_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)
        metrics = {'val_loss' : ret_metrics['loss']
                  }
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):      
        loss = torch.tensor(step_outputs).mean()
        return {'loss' : loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 2e-4, weight_decay = 0)

        import pl_bolts
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.args.trainer_max_epochs)
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1, 'monitor': 'val_loss', 'strict': True, 'name': 'WarmupCosineAnnealing' }
        return {'optimizer' : optimizer, 'lr_scheduler' : scheduler_config}


    def get_dataloader(self, args):
        data_directory = f'{args.dataset_root}/{args.dataset_celltype}'
        input_tracklist = [('ctcf_log2fc.bw', None), ('atac.bw','log')]  ## {('ctcf_log2fc.bw', None), ('atac.bw','log')}

        train_subset = GenomeDataset(data_directory, input_tracklist, model=args.pred_mode, mode='train', fold = args.fold)
        val_subset = GenomeDataset(data_directory, input_tracklist, model=args.pred_mode, mode='validate', fold = args.fold)
        test_subset = GenomeDataset(data_directory, input_tracklist, model=args.pred_mode, mode='test', fold = args.fold)

        # Define batch size and number of workers per GPU
        # gpus = args.trainer_num_gpu
        # batch_size = int(args.dataloader_batch_size / gpus)
        # num_workers = int(args.dataloader_num_workers / gpus)
        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers
                          
        # Create DataLoaders
        trainloader = torch.utils.data.DataLoader(
            train_subset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
    
        validateloader = torch.utils.data.DataLoader(
            val_subset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
    
        testloader = torch.utils.data.DataLoader(
            test_subset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
    
        return trainloader, validateloader, testloader


class TrainModule4M(TrainModule):
    def __init__(self, args, backgd1k, backgd2k, backgd4k, backgd8k, backgd16k):
        super(TrainModule4M, self).__init__(args)
        self.model = my_models.SwinT4M(2, mid_hidden = 128)
        if( args.ckpt_path is not None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            model_weights = checkpoint['state_dict']
            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            self.model.load_state_dict(model_weights)
            self.model.eval()
        self.args = args
        self.save_hyperparameters()
        self.backgd1k = backgd1k
        self.backgd2k = backgd2k
        self.backgd4k = backgd4k
        self.backgd8k = backgd8k
        self.backgd16k = backgd16k

    def forward(self, x, r8, r4, r2, r1):
        return self.model(x, r8, r4, r2, r1)

    def _shared_step(self, batch, batch_idx):
        inputs, mat1k = self.proc_batch(batch)
        mat1k = torch.nan_to_num(mat1k, nan=0.0)
        ## mat1k = torch.log(mat + 1)
        # mat1k = torch.nan_to_num(mat1k, nan=0.0)
        # mat1k = torch.log(torch.nan_to_num(mat1k.detach(), nan=0.0) *1000.0 +1)  

        mask1k = self.generate_mask(mat1k.detach())      
        if self.check_if_invalid(mask1k):
            # print(mat1k.shape)
            print(f"invalid input mat at: {batch_idx} ")
            # my_mean = mat1k.float().mean().item()  
            # self.file.write(f'Step {batch_idx}: Mean={my_mean}\n')  
            # self.file.flush()  #
            # torch.save(mat1k, 'tmp_mat.pt')
            # torch.save(mask1k, 'tmp_mask.pt')
            # torch.save(inputs, 'tmp_inputs.pt')
            loss = torch.tensor(0.0, requires_grad=True, device=mask1k.device)  
            return loss
        
        mat2k = self.downsample_tensor(mat1k.detach())
        mat4k = self.downsample_tensor(mat2k.detach())
        mat8k = self.downsample_tensor(mat4k.detach())
        mat16k = self.downsample_tensor(mat8k.detach())

        mask2k = self.downsample_mask(mask1k.float().detach(), 2)
        mask4k = self.downsample_mask(mask1k.float().detach(), 4)
        mask8k = self.downsample_mask(mask1k.float().detach(), 8)
        mask16k = self.downsample_mask(mask1k.float().detach(), 16)

        mat8knew, mask8knew, r8, start8 = self.sample_tensor(mat8k, mask8k, 0)
        mat4knew, mask4knew, r4, start4 = self.sample_tensor(mat4k, mask4k, start8) 
        mat2knew, mask2knew, r2, start2 = self.sample_tensor(mat2k, mask2k, start4) 
        mat1knew, mask1knew, r1, start1 = self.sample_tensor(mat1k, mask1k, start2) 

        out16k, out8k, out4k, out2k, out1k = self(inputs, r8, r4, r2, r1)   ## 
        
        eye_tensor = torch.eye(256, 256, dtype=torch.bool, device=mask16k.device).unsqueeze(0) 
        mask16k = self.diagonalize_mask(mask16k, ~eye_tensor)
        mask8knew = self.diagonalize_mask(mask8knew, ~eye_tensor)
        mask4knew = self.diagonalize_mask(mask4knew, ~eye_tensor)
        mask2knew = self.diagonalize_mask(mask2knew, ~eye_tensor)
        mask1knew = self.diagonalize_mask(mask1knew, ~eye_tensor)

        min1k = torch.min(self.backgd1k)
        min2k = torch.min(self.backgd2k)
        min4k = torch.min(self.backgd4k)
        min8k = torch.min(self.backgd8k)
        min16k = torch.min(self.backgd16k)

        bs = mask16k.shape[0]
        backgd16k = self.backgd16k.repeat(bs, 1, 1)
        backgd8k = self.backgd8k.repeat(bs, 1, 1)
        backgd4k = self.backgd4k.repeat(bs, 1, 1)
        backgd2k = self.backgd2k.repeat(bs, 1, 1)
        backgd1k = self.backgd1k.repeat(bs, 1, 1)

        criterion = torch.nn.MSELoss()
        loss16 = criterion(out16k[mask16k], torch.log(((mat16k[mask16k] + min16k) / (backgd16k[mask16k] + min16k)))) # criterion(out16k[mask16knew], torch.log(mat16knew[mask16knew]+1)) 
        loss8 = criterion(out8k[mask8knew], torch.log(((mat8knew[mask8knew] + min8k) / (backgd8k[mask8knew] + min8k)))) # criterion(out8k[mask8knew], torch.log(mat8knew[mask8knew]+1)) 
        loss4 = criterion(out4k[mask4knew], torch.log(((mat4knew[mask4knew] + min4k) / (backgd4k[mask4knew] + min4k)))) # criterion(out4k[mask4knew], torch.log(mat4knew[mask4knew]+1)) 
        loss2 = criterion(out2k[mask2knew], torch.log(((mat2knew[mask2knew] + min2k) / (backgd2k[mask2knew] + min2k)))) # criterion(out2k[mask2knew], torch.log(mat2knew[mask2knew]+1)) 
        loss1 = criterion(out1k[mask1knew], torch.log(((mat1knew[mask1knew] + min1k) / (backgd1k[mask1knew] + min1k)))) # criterion(out1k[mask1knew], torch.log(mat1knew[mask1knew]+1)) 
        loss = loss16 + loss8  + loss4 + loss2 + loss1
        return loss



class TrainModule32M(TrainModule):
    def __init__(self, args, backgd16k, backgd32k, backgd64k, backgd128k):
        super(TrainModule32M, self).__init__(args)
        self.model = my_models.SwinT32M(2, mid_hidden = 128)
        if( args.ckpt_path is not None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            checkpoint = torch.load(args.ckpt_path, map_location=device)
            model_weights = checkpoint['state_dict']
            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            self.model.load_state_dict(model_weights)
            self.model.eval()
        self.model.replace_with_predicts(args.preckpt_path)
        self.model.set_mode()  # Set the mode (eval/train) appropriately
        # check
        for name, module in self.model.named_modules():  
            if name in ['encoder', 'swin1d']:  
                print(f"{name} is in eval mode: {not module.training}")
        self.args = args
        self.save_hyperparameters()
        self.backgd16k = backgd16k
        self.backgd32k = backgd32k
        self.backgd64k = backgd64k
        self.backgd128k = backgd128k

    def forward(self, x, r64, r32, r16):
        return self.model(x, r64, r32, r16)

    def training_step(self, batch, batch_idx):  
        self.model.set_mode()  # Set the mode (eval/train) appropriately
        loss = self._shared_step(batch, batch_idx)  
        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, prog_bar=True)  ## batch_size = inputs.shape[0], 
        return loss
    def _shared_step(self, batch, batch_idx):
        inputs, mat16k = self.proc_batch(batch)
        mat16k = torch.nan_to_num(mat16k, nan=0.0)

        mask16k = self.generate_mask(mat16k.detach())
        if self.check_if_invalid(mask16k):      
            print(mat16k.shape)
            print(f"invalid input mat at: {batch_idx} ")
            # print(mask16k.shape)
            loss = torch.tensor(0.0, requires_grad=True, device=torch.device("cuda:0"))  
            return loss
        
        mat32k = self.downsample_tensor(mat16k.detach())
        mat64k = self.downsample_tensor(mat32k.detach())
        mat128k = self.downsample_tensor(mat64k.detach())

        mask32k = self.downsample_mask(mask16k.float().detach(), 2)
        mask64k = self.downsample_mask(mask16k.float().detach(), 4)
        mask128k = self.downsample_mask(mask16k.float().detach(), 8)

        mat64knew, mask64knew, r64, start = self.sample_tensor(mat64k, mask64k, 0) 
        mat32knew, mask32knew, r32, start = self.sample_tensor(mat32k, mask32k, start) 
        mat16knew, mask16knew, r16, start = self.sample_tensor(mat16k, mask16k, start) 

        out128k, out64k, out32k, out16k = self(inputs, r64, r32, r16)   ##  
        
        mask128k = self.diagonalize_mask2(mask128k)
        mask64knew = self.diagonalize_mask2(mask64knew)
        mask32knew = self.diagonalize_mask2(mask32knew)
        mask16knew = self.diagonalize_mask2(mask16knew)

        backgd16k = self.backgd16k.to(out16k.device)  
        backgd32k = self.backgd32k.to(out32k.device)  
        backgd64k = self.backgd64k.to(out64k.device)  
        backgd128k = self.backgd128k.to(out128k.device)  
        min16 = torch.min(backgd16k)
        min32 = torch.min(backgd32k)
        min64 = torch.min(backgd64k)
        min128 = torch.min(backgd128k)

        bs = mask128k.shape[0]
        backgd128k = backgd128k.repeat(bs, 1, 1)
        backgd64k = backgd64k.repeat(bs, 1, 1)
        backgd32k = backgd32k.repeat(bs, 1, 1)
        backgd16k = backgd16k.repeat(bs, 1, 1)

        criterion = torch.nn.MSELoss()
        loss128 = criterion(out128k[mask128k], torch.log(((mat128k[mask128k] + min128) / (backgd128k[mask128k] + min128)))) 
        loss64 = criterion(out64k[mask64knew], torch.log(((mat64knew[mask64knew] + min64) / (backgd64k[mask64knew] + min64)))) 
        loss32 = criterion(out32k[mask32knew], torch.log(((mat32knew[mask32knew] + min32) / (backgd32k[mask32knew] + min32))))
        loss16 = criterion(out16k[mask16knew], torch.log(((mat16knew[mask16knew] + min16) / (backgd16k[mask16knew] + min16)))) 
        loss = loss128  + loss64 + loss32 + loss16
        return loss    
if __name__ == '__main__':
    main()
