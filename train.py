
import warnings
from utils.mlflow_image_logging import MLFlowImageLogger
from train_val_functions import MapToLossFunction, ship_device, statistics
import network.end2end as network_utils

from typing import List
from tqdm import tqdm


import time
import torch
import torch.nn as nn
import os
import mlflow
import wandb

# ADDITIONAL CODE: get run from the current context
#run = Run.get_context()

           

class RoboFishNet():
    
    def __init__(self, ver_name, unordered:bool, spatial_data:bool, merfish_data:bool, img_dir_timestamp:str,
                 groundtruth_emitters_count, scheduler, optimizer_fn, in_networks:int, 
                 in_channels:int, outputs:List, gpus:int, num_tiles:int, batch_size:int, 
                 lr:int, clip_value:int, network_opt:int, mcce_loss_opt:int, 
                 loss_scale:List, checkpoint_dir:str, delta_loss=0.01, epochs=2, print_rate=260):

        super().__init__()
        
        self.scheduler = scheduler 
        
        self.optimizer_fn = optimizer_fn
                                
        self.outputs = outputs
        
        self.epochs = epochs
        
        self.print_rate = print_rate
        
        self.checkpoint_dir = checkpoint_dir
        
        self.lr = lr
        
        self.clip_value = clip_value
        
        if torch.cuda.is_available() and gpus>0:
            
            self.target = torch.cuda.current_device()
            
        else:
            
            self.target = 'cpu'
        
        self.th1 = 0.6
        
        self.th2 = 0.9
        
        self.delta_loss = delta_loss

        self.total_classes = outputs
        
        self.batch_size = ship_device(torch.tensor(batch_size), self.target)
        
        self.mcce_loss_opt = mcce_loss_opt
                
        self.loss_scale = loss_scale
                                        
        self.mlflow_image_logger = MLFlowImageLogger(self.th1, self.th2, self.target)
        
        self.train_eval_step_num = int((num_tiles*0.7)/batch_size)
        
        self.val_eval_step_num = int((num_tiles*0.3)/batch_size)

        
        mlflow.log_params({"optimizer_version_name": ver_name,
                           "unordered_data": unordered,
                           "spatial_data": spatial_data,
                           "merfish_data": merfish_data,
                           "dataset": img_dir_timestamp,
                           "groundtruth_emitters_count": groundtruth_emitters_count,
                           "mcce_loss_opt": mcce_loss_opt,
                           "learning_rate_mcce": lr,
                           "epochs": epochs,
                           "clipping_norm": clip_value,
                           "postprocess_th1": self.th1,
                           "postprocess_th2": self.th2,
                           "mcce_loss_weight": loss_scale,
                           "output_num": outputs,
                           "total_classes_count": outputs,
                           "codebook_gene_count": outputs - 1,
                           "delta_loss": delta_loss
                           })
        
        wandb.config.update({"optimizer_version_name": ver_name,
                             "unordered_data": unordered,
                             "spatial_data": spatial_data,
                             "merfish_data": merfish_data,
                             "dataset": img_dir_timestamp,
                             "groundtruth_emitters_count": groundtruth_emitters_count,
                             "mcce_loss_opt": mcce_loss_opt,
                             "learning_rate_mcce": lr,
                             "epochs": epochs,
                             "clipping_norm": clip_value,
                             "postprocess_th1": self.th1,
                             "postprocess_th2": self.th2,
                             "mcce_loss_weight": loss_scale,
                             "output_num": outputs,
                             "total_classes_count": outputs,
                             "codebook_gene_count": outputs - 1,
                             "delta_loss": delta_loss
                             })

        
        
    def __call__(self, model, train_dataloader, val_dataloader):
    
        val_lowest = 99999999
        
        val_seg_lowest = 99999999
        
        val_dec_lowest = 99999999
                
        for e in range(self.epochs):
            
            train_step_count = 0
            
            epoch_loss = 0
            
            epoch_seg_loss = 0
            
            epoch_dec_loss = 0
                
            model.train()
            
            weight_loss_seg, weight_loss_dec = network_utils.get_loss_weights(self.delta_loss, e, self.epochs)
            
            # Always return 0, since we assume gradient_adjustment = True
            dec_gradient_multiplier = network_utils.get_dec_gradient_multiplier()
            
            network_utils.set_dec_gradient_multiplier(model, dec_gradient_multiplier)

            
            for x, y in tqdm(train_dataloader):
                    
                x, groundtruth_labels_and_barcode = ship_device([x[:,0,:,:,:], y[:,0,:,:,:]], self.target)
                
                groundtruth_labels = groundtruth_labels_and_barcode[:, 0, :, :]
                
                train_step_count += 1
                
                self.optimizer_fn.zero_grad()
                                
                classification_map, seg_mask = model(x)
                
                train_loss, train_seg_loss, train_dec_loss = \
                    MapToLossFunction(self.target, self.mcce_loss_opt, classification_map, 
                                      seg_mask, groundtruth_labels_and_barcode, weight_loss_seg, weight_loss_dec)
                

                """ Calling loss backward """
                train_loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_value, norm_type=2)
                
                self.optimizer_fn.step()
                
                epoch_loss += train_loss.item()
                
                epoch_seg_loss += train_seg_loss.item()
                
                epoch_dec_loss += train_dec_loss.item()



                """Print Update and Log image visualizations"""
                if train_step_count % self.train_eval_step_num == 0:
                    
                    train_denom = train_step_count
                    print('Epoch: {} | Avg Training Loss per {} steps: {}'.format(e, self.train_eval_step_num, epoch_loss/train_denom))
    

                if train_step_count == self.train_eval_step_num:
                    
                    """Generate Statistics: Losses, Scores/Accuracy, Emitter Counts at the end of Train Epoch"""
                    statistics(self.batch_size, classification_map, groundtruth_labels, e, 'train')
            
                    
                    if e%10 == 0:

                        """Visualize Train Images at the end of Train Epoch"""
                        self.mlflow_image_logger(classification_map[0], groundtruth_labels[0], self.total_classes, e, 'train')
                    

            train_denom = train_step_count
            
            mlflow.log_metric('train_loss', epoch_loss/train_denom)
            mlflow.log_metric('train_seg_loss', epoch_seg_loss/train_denom)
            mlflow.log_metric('train_dec_loss', epoch_dec_loss/train_denom)

            
            wandb.log({'train_loss': epoch_loss/train_denom,
                       'epoch': e,
                       'train_seg_loss': epoch_seg_loss/train_denom,
                       'train_dec_loss': epoch_dec_loss/train_denom})


    
            """ Turn on model evaluation mode."""
            model.eval()
            
            val_step_count = 0
            
            total_val_loss = 0
            
            total_seg_loss = 0
            
            total_dec_loss = 0
            
            avg_ji, avg_pre, avg_rec = 0, 0, 0
            
            for x, y in tqdm(val_dataloader):
                                
                x, groundtruth_labels_and_barcode = ship_device([x[:,0,:,:,:], y[:,0,:,:,:]], self.target)
                
                groundtruth_labels = groundtruth_labels_and_barcode[:, 0, :, :]
                                
                val_step_count += 1
                
                classification_map, seg_mask = model(x)
                
                val_loss, val_seg_loss, val_dec_loss = \
                    MapToLossFunction(self.target, self.mcce_loss_opt, classification_map, 
                                      seg_mask, groundtruth_labels_and_barcode, weight_loss_seg, weight_loss_dec)
                
                total_val_loss += val_loss.item()
                
                total_seg_loss += val_seg_loss.item()
                
                total_dec_loss += val_dec_loss.item()
                
                
                if val_step_count == self.val_eval_step_num:
            
                    """Generate Statistics: Losses, Scores/Accuracy, Emitter Counts at the end of Val Epoch"""
                    avg_ji, avg_pre, avg_rec = statistics(self.batch_size, classification_map, groundtruth_labels, e, 'val')


                    if e%10 == 0:
                    
                        """Visualize Val Images at the end of Val Epoch"""
                        self.mlflow_image_logger(classification_map[0], groundtruth_labels[0], self.total_classes, e, 'val')
    
    
            val_denom = val_step_count
            
            val_loss = total_val_loss/val_denom
            
            val_seg_loss = total_seg_loss/val_denom
            
            val_dec_loss = total_dec_loss/val_denom
            
            mlflow.log_metric('val_loss', val_loss)            
            mlflow.log_metric('val_seg_loss', val_seg_loss)
            mlflow.log_metric('val_dec_loss', val_dec_loss)

            
            wandb.log({'val_loss': val_loss,
                       'epoch': e,
                       'val_seg_loss': val_seg_loss,
                       'val_dec_loss': val_dec_loss})

            
            
            if val_loss < val_lowest or val_seg_loss < val_seg_lowest or val_dec_loss < val_dec_lowest:
                
                best_model = model
                
                #mlflow.pytorch.log_state_dict(best_model.state_dict(), 
                #                              artifact_path=os.path.join(self.checkpoint_dir, f"{time.time()}_epoch{e}")
                #                              )
                
                try:
                    model_path = os.path.join(wandb.run.dir, f'model_epoch_{e}.pth')
                    
                    torch.save(best_model.state_dict(), model_path)
                    
                    artifact = wandb.Artifact('model', type='model')

                    artifact.add_file(model_path)
                    
                    wandb.log_artifact(artifact)

                    mlflow.pytorch.log_model(best_model, f'model_epoch_{e}')
                    
                except:
                    
                    print('PermissionError: [WinError 32] The process cannot access the file because it is being used by another process.')
                    print('Proceeding with updating best metrics and lowest validation loss.')
                
                if val_loss < val_lowest:
                
                    val_lowest = val_loss

                if val_seg_loss < val_seg_lowest:
                    
                    val_seg_lowest = val_seg_loss
                    
                if val_dec_loss < val_dec_lowest:
                    
                    val_dec_lowest = val_dec_loss

                best_ji, best_precision, best_recall = avg_ji, avg_pre, avg_rec
            
            # https://pytorch.org/docs/stable/optim.html
            self.scheduler.step(val_seg_loss, val_dec_loss)
            
        return best_model, best_ji, best_precision, best_recall



