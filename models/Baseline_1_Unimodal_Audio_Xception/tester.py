import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
from utils import system_util, train_util

# Call logger for monitoring (in terminal)
logger = system_util.get_logger(__name__)


class Tester(object):
    def __init__(self, config, model, datasets, dataloaders, criterion, optimizer, scheduler, gpuid, wandb_run):
        
        ''' Default setting '''
        self.config = config
        self.gpuid = gpuid[0]
        self.location = 'cuda:{}'.format(self.gpuid)
        self.device = torch.device(self.location)
        self.model = model.to(self.device)
        
        train_util.print_parameters_count(self.model)
        
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if any(file.endswith('.pt') for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "pretrain_weights"))):
            self.checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "pretrain_weights")
        else:
            self.checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "checkpoint")
        
        self.start_epoch = train_util.load_latent_checkpoint(self.checkpoint_path, self.model, self.optimizer, location='cuda:{}'.format(gpuid[0]))
        
        self.wandb_run = wandb_run

    def _test(self, dataloader):
        self.model.eval()
        test_loss = 0
        total_mfcc_correct, total_mfcc_num, avg_each_mfcc_acc = 0, 0, 0
        records = []
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="grey", dynamic_ncols=True)
        with torch.inference_mode():
            for idx, batch in enumerate(dataloader):
                B, T, C, H, W = batch["mfcc_data"].size() # torch.Size([Batch, T=clips, C, H, W])
                mfccs =  batch["mfcc_data"].contiguous().permute(1, 0, 2, 3, 4) # torch.Size([T, B, C, H, W])
                mfcc_labels = batch["mfcc_class"] # torch.Size([B, T])
                mfcc_path = batch["mfcc_path"][0]
                current_mfcc_correct = 0

                ### Move Data to Device (Ideally GPU)
                mfccs = mfccs.to(self.device)
                mfcc_labels = mfcc_labels.to(self.device)
                record_label = mfcc_labels[0].tolist()
                preds = []
                for idx in range(len(mfccs)):
                    ### Forward Propagation
                    mfcc = mfccs[idx]
                    mfcc_label = mfcc_labels[0][idx]
                    outputs = self.model(mfcc)
                    loss = self.criterion(outputs, mfcc_label.unsqueeze(dim=0))
                    
                    current_mfcc_correct += int((torch.argmax(outputs, axis=1) == mfcc_label).sum())
                    test_loss += loss.item()
                    preds.append(torch.argmax(outputs, axis=1))
                
                current_mfcc_acc = (current_mfcc_correct / (B * T)) * 100
                avg_each_mfcc_acc += current_mfcc_acc
                total_mfcc_correct += current_mfcc_correct
                total_mfcc_num += (B * T)
                
                # Append results to the records list
                for mfcc_path, mfcc_acc in zip([mfcc_path], [current_mfcc_acc]):
                    name = os.path.basename(mfcc_path)
                    label = record_label
                    pred = [pred.item() for pred in preds]
                    boolean = [a == b for a, b in zip(pred, label)]
                    co_num = boolean.count(True)
                    to_num = len(boolean)
                    acc = mfcc_acc
                    records.append({
                        "audio_name": name,
                        "audio_label": label,
                        "audio_pred": pred,
                        "boolean": boolean,
                        "correct_num": co_num,
                        "total_num": to_num,
                        "audio_acc": acc
                    })
                
                pbar.set_postfix(loss="{:.08f}".format(float(test_loss / (B * T * (idx + 1)))),
                                acc="{:.04f}%".format(current_mfcc_acc))
                pbar.update()
            pbar.close()
            
            df = pd.DataFrame(records)
            df.to_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "each_file_record.xlsx"), index=False)
            
            test_loss = test_loss / (B * T * len(dataloader))
            avg_each_mfcc_acc /= len(dataloader)
            total_mfcc_acc = total_mfcc_correct / total_mfcc_num * 100
        
        return test_loss, avg_each_mfcc_acc, total_mfcc_acc

    def run(self):
        torch.cuda.empty_cache()
        gc.collect()
        with torch.cuda.device(self.gpuid):
            self.wandb_run.watch(self.model, log="all")
            
            test_loss, avg_each_mfcc_acc, total_mfcc_acc = self._test(self.dataloaders['test'])

            logger.info("\t Test loss: {:.08f}".format(test_loss))
            logger.info("\t Average each mfcc accuracy: {:.04f}".format(avg_each_mfcc_acc))
            logger.info("\t Total mfcc accuracy: {:.04f}".format(total_mfcc_acc))

            # Log metrics at each epoch using wandb
            self.wandb_run.log({'Test loss': test_loss})
            self.wandb_run.log({'Average each mfcc accuracy': avg_each_mfcc_acc})
            self.wandb_run.log({'Total mfcc accuracy': total_mfcc_acc})