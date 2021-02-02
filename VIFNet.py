import logging
import os, sys
sys.path.append('/data/yujsh/xiaoxiannv/fusion')

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse

from data_utils.TNODataset import TNODataset
from model.VIFNet_resnet_v2 import VIFNet_resnet_v2
from trainer.Train import Train

# log format
C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, 
                        help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=2e-5, 
                        help='learning rate')
parser.add_argument('--device_id', type=int, default=0,
                        help='device number')
parser.add_argument('--model_name', type=str, default='test',
                        help='model_name')

args = parser.parse_args()

device_id = args.device_id 
use_cuda = args.device_id != -1
batch_size = args.batch_size
model_name = args.model_name


data_path = '/data/yujsh/xiaoxiannv/fusion/dataset/vif_dataset'
if not os.path.exists(data_path):
    log.error(f"data path {data_path} didn't exitis")

if __name__ == '__main__':
    dataset = TNODataset(data_path, device_id=device_id)
    train_size = int(0.8 * len(dataset))
    dev_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size * 5), shuffle=True)

    logging.info(f"train data all steps: {len(train_loader)}, validate data all steps : {len(val_loader)}")

    model = VIFNet_resnet_v2(device_id)
    if use_cuda:
        model = model.cuda(device_id)

    # Prepare  optimizer and schedule(linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    trainer = Train(model_name=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=None,
                    model=model,
                    optimizer=optimizer,
                    epochs=100,
                    print_step=1,
                    early_stop_patience=3,
                    save_model_path=f'./save_model/{model_name}',
                    save_model_every_epoch=True,
                    tensorboard_path=f'./save_model/{model_name}')
    print(trainer.train())

