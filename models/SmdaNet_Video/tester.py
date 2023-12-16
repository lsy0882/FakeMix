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
        
        # self.start_epoch = train_util.load_latent_checkpoint(self.checkpoint_path, self.model, self.optimizer, location='cuda:{}'.format(gpuid[0]))
        
        self.wandb_run = wandb_run
            
    def _test(self, dataloader):
        self.model.eval()
        test_loss = 0
        total_video_correct, total_video_num, avg_each_video_acc = 0, 0, 0
        records = []
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="grey", dynamic_ncols=True)
        with torch.inference_mode():
            for idx, batch in enumerate(dataloader):
                B, T, F, C, H, W = batch["video_data"].size() # torch.Size([Batch, Time, Frame, Channel, Height, Width])
                videos =  batch["video_data"].contiguous().permute(1, 2, 0, 3, 4, 5) # torch.Size([T, F, B, C, H, W])
                videos = videos[:, 12, :, :, :, :] # torch.Size([T, B, C, H, W])
                videos_label = batch["video_class"] # torch.Size([B, T])
                video_path = batch["video_path"][0]
                current_video_correct = 0
                
                ### Move Data to Device (Ideally GPU)
                videos = videos.to(self.device)
                videos_label = videos_label.to(self.device)
                video_record_label = videos_label[0].tolist()
                video_preds = []
                
                for i in range(T):
                    if i == 0:
                        continue
                    ### Forward Propagation
                    previous_video = videos[i-1]
                    current_video = videos[i]
                    video_pair = torch.cat((previous_video, current_video), dim=0).unsqueeze(dim=0)
                    current_video_label = videos_label[0][i]

                    outputs = self.model(video_pair) # outputs = tuple(weigted_v_softmax, weigted_a_softmax)
                    v_loss = self.criterion(outputs, current_video_label.unsqueeze(dim=0))
                        
                    current_video_correct += int((torch.argmax(outputs, axis=1) == current_video_label).sum())
                    test_loss += v_loss.item()
                    
                    video_preds.append(torch.argmax(outputs, axis=1))
                
                current_video_acc = (current_video_correct / (B * T)) * 100
                avg_each_video_acc += current_video_acc
                total_video_correct += current_video_correct
                total_video_num += (B * T)
                
                
                # Append results to the records list
                for video_path, video_acc in zip([video_path], [current_video_acc]):
                    v_name = os.path.basename(video_path)
                    v_label = video_record_label[1:]
                    v_pred = [video_pred.item() for video_pred in video_preds]
                    v_boolean = [a == b for a, b in zip(v_pred, v_label)]
                    v_co_num = v_boolean.count(True)
                    v_to_num = len(v_boolean)
                    v_acc = video_acc
                    
                    
                    records.append({
                        "video_name": v_name,
                        "video_label": v_label,
                        "video_pred": v_pred,
                        "video_boolean": v_boolean,
                        "video_correct_num": v_co_num,
                        "video_total_num": v_to_num,
                        "video_acc": v_acc
                    })

                pbar.set_postfix(loss="{:.08f}".format(float(test_loss / (B * T * (idx + 1)))),
                                v_acc="{:.04f}%".format(current_video_acc))
                pbar.update()
            pbar.close()
            
            df = pd.DataFrame(records)
            df.to_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "each_file_record.xlsx"), index=False)
            
            test_loss = test_loss / (B * T * len(dataloader))
            avg_each_video_acc /= len(dataloader)
            total_video_acc = total_video_correct / total_video_num * 100

        return test_loss, avg_each_video_acc, total_video_acc

    def run(self):
        torch.cuda.empty_cache()
        gc.collect()
        with torch.cuda.device(self.gpuid):
            self.wandb_run.watch(self.model, log="all")
            
            test_loss, avg_each_video_acc, total_video_acc= self._test(self.dataloaders['test'])
            
            logger.info("\t Test loss: {:.08f}".format(test_loss))
            logger.info("\t Average each video accuracy: {:.04f}".format(avg_each_video_acc))
            logger.info("\t Total video accuracy: {:.04f}".format(total_video_acc))

            # Log metrics at each epoch using wandb
            self.wandb_run.log({'Test loss': test_loss})
            self.wandb_run.log({'Average each video accuracy': avg_each_video_acc})
            self.wandb_run.log({'Total video accuracy': total_video_acc})