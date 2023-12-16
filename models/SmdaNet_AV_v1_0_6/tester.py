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
        total_mfcc_correct, total_mfcc_num, avg_each_mfcc_acc = 0, 0, 0
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
                
                mfccs =  batch["mfcc_data"].contiguous().permute(1, 0, 2, 3, 4) # torch.Size([T, B, C, H, W])
                mfcc_labels = batch["mfcc_class"] # torch.Size([B, T])
                mfcc_path = batch["mfcc_path"][0]
                current_mfcc_correct = 0
                
                ### Move Data to Device (Ideally GPU)
                videos = videos.to(self.device)
                videos_label = videos_label.to(self.device)
                video_record_label = videos_label[0].tolist()
                video_preds = []
                mfccs = mfccs.to(self.device)
                mfcc_labels = mfcc_labels.to(self.device)
                audio_record_label = mfcc_labels[0].tolist()
                audio_preds = []
                
                for i in range(T):
                    if i == 0:
                        continue
                    ### Forward Propagation
                    previous_video = videos[i-1]
                    current_video = videos[i]
                    video_pair = torch.cat((previous_video, current_video), dim=0).unsqueeze(dim=0)
                    current_video_label = videos_label[0][i]
                    previous_mfcc = mfccs[i-1]
                    current_mfcc = mfccs[i]
                    mfcc_pair = torch.cat((previous_mfcc, current_mfcc), dim=0).unsqueeze(dim=0)
                    current_mfcc_label = mfcc_labels[0][i]

                    outputs = self.model(video_pair, mfcc_pair) # outputs = tuple(weigted_v_softmax, weigted_a_softmax)
                    v_loss = self.criterion(outputs[0], current_video_label.unsqueeze(dim=0))
                    a_loss = self.criterion(outputs[1], current_mfcc_label.unsqueeze(dim=0))
                    v_a_loss = v_loss + a_loss
                        
                    current_video_correct += int((torch.argmax(outputs[0], axis=1) == current_video_label).sum())
                    current_mfcc_correct += int((torch.argmax(outputs[1], axis=1) == current_mfcc_label).sum())
                    test_loss += v_a_loss.item()
                    
                    video_preds.append(torch.argmax(outputs[0], axis=1))
                    audio_preds.append(torch.argmax(outputs[1], axis=1))
                
                current_video_acc = (current_video_correct / (B * T)) * 100
                avg_each_video_acc += current_video_acc
                total_video_correct += current_video_correct
                total_video_num += (B * T)
                
                current_mfcc_acc = (current_mfcc_correct / (B * T)) * 100
                avg_each_mfcc_acc += current_mfcc_acc
                total_mfcc_correct += current_mfcc_correct
                total_mfcc_num += (B * T)
                
                # Append results to the records list
                for video_path, video_acc, audio_path, audio_acc in zip([video_path], [current_video_acc], [mfcc_path], [current_mfcc_acc]):
                    v_name = os.path.basename(video_path)
                    v_label = video_record_label[1:]
                    v_pred = [video_pred.item() for video_pred in video_preds]
                    v_boolean = [a == b for a, b in zip(v_pred, v_label)]
                    v_co_num = v_boolean.count(True)
                    v_to_num = len(v_boolean)
                    v_acc = video_acc
                    
                    a_name = os.path.basename(audio_path)
                    a_label = audio_record_label[1:]
                    a_pred = [audio_pred.item() for audio_pred in audio_preds]
                    a_boolean = [a == b for a, b in zip(a_pred, a_label)]
                    a_co_num = a_boolean.count(True)
                    a_to_num = len(a_boolean)
                    a_acc = audio_acc
                    
                    records.append({
                        "video_name": v_name,
                        "video_label": v_label,
                        "video_pred": v_pred,
                        "video_boolean": v_boolean,
                        "video_correct_num": v_co_num,
                        "video_total_num": v_to_num,
                        "video_acc": v_acc,
                        "audio_name": a_name,
                        "audio_label": a_label,
                        "audio_pred": a_pred,
                        "audio_boolean": a_boolean,
                        "audio_correct_num": a_co_num,
                        "audio_total_num": a_to_num,
                        "audio_acc": a_acc
                    })

                pbar.set_postfix(loss="{:.08f}".format(float(test_loss / (B * T * (idx + 1)))),
                                v_acc="{:.04f}%".format(current_video_acc),
                                a_acc="{:.04f}%".format(current_mfcc_acc))
                pbar.update()
            pbar.close()
            
            df = pd.DataFrame(records)
            df.to_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), "each_file_record.xlsx"), index=False)
            
            test_loss = test_loss / (B * T * len(dataloader))
            avg_each_video_acc /= len(dataloader)
            total_video_acc = total_video_correct / total_video_num * 100
            avg_each_mfcc_acc /= len(dataloader)
            total_mfcc_acc = total_mfcc_correct / total_mfcc_num * 100

        return test_loss, avg_each_video_acc, total_video_acc, avg_each_mfcc_acc, total_mfcc_acc

    def run(self):
        torch.cuda.empty_cache()
        gc.collect()
        with torch.cuda.device(self.gpuid):
            self.wandb_run.watch(self.model, log="all")
            
            test_loss, avg_each_video_acc, total_video_acc, avg_each_mfcc_acc, total_mfcc_acc = self._test(self.dataloaders['test'])
            
            logger.info("\t Test loss: {:.08f}".format(test_loss))
            logger.info("\t Average each video accuracy: {:.04f}".format(avg_each_video_acc))
            logger.info("\t Total video accuracy: {:.04f}".format(total_video_acc))
            logger.info("\t Average each audio accuracy: {:.04f}".format(avg_each_mfcc_acc))
            logger.info("\t Total audio accuracy: {:.04f}".format(total_mfcc_acc))

            # Log metrics at each epoch using wandb
            self.wandb_run.log({'Test loss': test_loss})
            self.wandb_run.log({'Average each video accuracy': avg_each_video_acc})
            self.wandb_run.log({'Total video accuracy': total_video_acc})
            self.wandb_run.log({'Average each audio accuracy': avg_each_mfcc_acc})
            self.wandb_run.log({'Total audio accuracy': total_mfcc_acc})