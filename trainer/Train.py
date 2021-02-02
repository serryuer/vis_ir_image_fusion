import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import *
# log format
from tensorboardX import SummaryWriter
from tqdm import tqdm

C_LogFormat = '%(asctime)s - %(levelname)s - %(message)s'
# setting log format
logging.basicConfig(level=logging.INFO, format=C_LogFormat)

class Train(object):
    def __init__(self, model_name, train_loader, val_loader, test_loader,  model, optimizer, epochs, print_step,
                 early_stop_patience, save_model_path, save_model_every_epoch=True, tensorboard_path=None,
                 test_every_epoch=False, test_images_path=None, test_result_path=None, image_channel=1):
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.print_step = print_step
        self.early_stop_patience = early_stop_patience
        self.save_model_every_epoch = save_model_every_epoch
        self.save_model_path = save_model_path
        self.tensorboard_path = tensorboard_path

        if not os.path.isdir(self.save_model_path):
            os.makedirs(self.save_model_path)
        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.best_val_epoch = 0
        self.best_val_loss = 100000

        self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path)
        self.test_every_epoch = test_every_epoch
        self.test_images_path = test_images_path
        self.test_result_path = test_result_path

        self.image_channel = image_channel

    def _save_model(self, model_name):
        torch.save(self.model, os.path.join(self.save_model_path, model_name + '.pt'))

    def _early_stop(self, epoch, loss):
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            self.best_val_epoch = epoch
            self._save_model(f'best-validate-model-{epoch}')
        else:
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + self.model_name + f"Validate has not promote {epoch - self.best_val_epoch}/{self.early_stop_patience}")
            if epoch - self.best_val_epoch > self.early_stop_patience:
                logging.info(self.model_name + f"-epoch {epoch}" + ":"
                             + f"Early Stop Train, best score locate on {self.best_val_epoch}, "
                             f"the best score is {self.best_val_loss}")
                return True
        return False

    def eval(self):
        logging.info(self.model_name + ":" + "## Start to evaluate. ##")
        self.model.eval()
        eval_loss = 0.0
        total_batch = 0
        for batch_count, batch_data in enumerate(self.val_loader):
            total_batch += 1
            with torch.no_grad():
                #batch_data = batch_data.cuda()
                outputs = self.model(batch_data)
                fuse_image, loss= outputs
                eval_loss += loss.mean().item()
                logging.info(self.model_name
                             + f"batch {batch_count + 1} : loss is {loss.mean().item()}")
        return eval_loss / total_batch

    def train(self):
        for epoch in range(self.epochs):
            tr_loss = 0.0
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"## The {epoch} Epoch, all {self.epochs} Epochs ! ##")
            logging.info(self.model_name + f"-epoch {epoch}" + ":"
                         + f"The current learning rate is {self.optimizer.param_groups[0].get('lr')}")
            self.model.train()
            since = time.time()
            for batch_count, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                fuse_image, loss= outputs
                loss = loss.sum()
                loss.backward()
                self.optimizer.step()
                if (batch_count + 1) % self.print_step == 0:
                    logging.info(self.model_name + f"-epoch {epoch}" + ":"
                                 + f"batch {batch_count + 1} : loss is {loss.mean().item()}")
                    self.tb_writer.add_scalar(f'{self.model_name}-scalar/train_loss', loss.mean().item(),
                                              batch_count + epoch * len(self.train_loader))

            if self.test_every_epoch:
                self.test_images(epoch)
            val_loss = self.eval()

            self.tb_writer.add_scalar(f'{self.model_name}-scalar/validate_loss', val_loss, epoch)

            logging.info(self.model_name + ": epoch" +
                         f" {epoch} Finished with time {format(time.time() - since)}, " +
                         f"validate loss {val_loss}")
            if self.save_model_every_epoch:
                self._save_model(f"{self.model_name}-{epoch}-{val_loss}")
            if self._early_stop(epoch, val_loss):
                break
        self.tb_writer.close()

    def test_images(self, epoch):
        if not os.path.exists(self.test_images_path):
            logging.error(f"{self.test_images_path} doesn't exits")
        if not os.path.exists(self.test_result_path):
            os.makedirs(self.test_result_path)
        images = os.listdir(os.path.join(self.test_images_path, 'IR'))
        from PIL import Image
        from torchvision import transforms
        for image in images:
            logging.info(f'==============testing image {image}==============')
            img_v = Image.open(os.path.join(os.path.join(self.test_images_path, 'IR'), image))
            img_r = Image.open(os.path.join(os.path.join(self.test_images_path, 'VIS'), image))
            if self.image_channel == 1:
                img_v = img_v.convert('L')
                img_r = img_r.convert('L')
            else:
                img_v = img_v.convert('RGB')
                img_r = img_r.convert('RGB')
            img_v_tensor = transforms.ToTensor()(img_v)
            img_r_tensor = transforms.ToTensor()(img_r)
            input = torch.stack([img_v_tensor, img_r_tensor]).unsqueeze(0)

            save_dir = os.path.join(self.test_result_path, f'epoch_{epoch}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with torch.no_grad():
                fuse_img_tensor, _ = self.model(input)
                fuse_img_tensor = fuse_img_tensor.cpu().squeeze(0)
                fuse_img = transforms.ToPILImage()(fuse_img_tensor)

                new_size = (img_v.size[0], img_v.size[1] * 3 + 60)
                total_image = Image.new('L', new_size)
                total_image.paste(img_v, (0, 0))
                total_image.paste(img_r, (0, img_r.size[1] + 30))
                total_image.paste(fuse_img, (0, img_r.size[1] * 2 + 60))

                total_image.save(os.path.join(save_dir, f'fusion_{image}.bmp'))

