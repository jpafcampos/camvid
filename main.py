from __future__ import print_function
import os
import os.path as osp
import copy
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import torch, os
from glob import glob
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
import vit
import resvit

from torch.optim import Adam
from torch.nn import CrossEntropyLoss

gpu_id = 2
lr = 0.01
wd = 0.0005
epochs = 700

import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


def validate(model, loader, device, metrics, save_val_results = False):
    """Do validation"""
    metrics.reset()

    if save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = U.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            try:
                outputs = outputs['out']
            except:
                pass
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(plt.ticker.NullLocator())
                    ax.yaxis.set_major_locator(plt.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score

class CamVidDataset(Dataset):

	def __init__(self, images, labels, height, width):
		self.images = images
		self.labels = labels
		self.height = height
		self.width = width
	
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		image_id = self.images[index]
		label_id = self.labels[index]
		# Read Image
		x = Image.open(image_id)
		x = [np.array(x)]
		x = np.stack(x, axis=2)
		x = torch.tensor(x).transpose(0, 2).transpose(1, 3) # Convert to N, C, H, W
		# Read Mask
		y = Image.open(label_id)
		y = [np.array(y)]
		y = torch.tensor(y)
		return x.squeeze(), y.squeeze()

def decode_segmap(image, color_dict):
    label_colours = np.array([
		color_dict['obj0'], color_dict['obj1'],
		color_dict['obj2'], color_dict['obj3'],
		color_dict['obj4'], color_dict['obj5'],
		color_dict['obj6'], color_dict['obj7'],
		color_dict['obj8'], color_dict['obj9'],
		color_dict['obj10'], color_dict['obj11']
	]).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, 12):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb

def predict_rgb(model, tensor, color_dict):
    with torch.no_grad():
        out = model(tensor.float()).squeeze(0)
    out = out.data.max(0)[1].cpu().numpy()
    return decode_segmap(out, color_dict)

color_dict = {
    'obj0' : [255, 0, 0], # Sky
    'obj1' : [0, 51, 204], # Building
    'obj2' : [0, 255, 255], # Posts
    'obj3' : [153, 102, 102], # Road
    'obj4' : [51, 0, 102], # Pavement
    'obj5' : [0, 255, 0], # Trees
    'obj6' : [102, 153, 153], # Signs
    'obj7' : [204, 0, 102], # Fence
    'obj8' : [102, 0, 0], # Car
    'obj9' : [0, 153, 102], # Pedestrian
    'obj10' : [255, 255, 255], # Cyclist
    'obj11' : [0, 0, 0] # bicycles
}

def get_class_weights(loader, num_classes, c=1.02):
    _, y= next(iter(loader))
    y_flat = y.flatten()
    each_class = np.bincount(y_flat, minlength=num_classes)
    p_class = each_class / len(y_flat)
    return 1 / (np.log(c + p_class))

def train(model, train_dataloader, val_dataloader,device, criterion, optimizer, train_step_size, val_step_size,visualize_every, save_every, save_location, save_prefix, epochs):
    metrics = 12
    # Make sure that the checkpoint location exists
    try:
    	os.mkdir(save_location)
    except:
    	pass
    train_loss_history, val_loss_history = [], []
    # Training
    for epoch in range(1, epochs + 1):
        print('Epoch {}\n'.format(epoch))
        # Training
        start = time()
        train_loss = 0
        model.train()
        # Step Loop
        for i in range(train_step_size):
        	x_batch, y_batch = next(iter(train_dataloader))
        	x_batch = x_batch.squeeze().to(device)
        	y_batch = y_batch.squeeze().to(device)
        	optimizer.zero_grad()
        	out = model(x_batch.float())
        	loss = criterion(out, y_batch.long())
        	loss.backward()
        	optimizer.step()
        	train_loss += loss.item()
        train_loss_history.append(train_loss / train_step_size)
        print('\nTraining Loss: {}'.format(train_loss_history[-1]))
        print('Training Time: {} seconds'.format(time() - start))
        # Validation
        val_loss = 0
        model.eval()   
        score = validate(model, val_dataloader, device, metrics)
        print(metrics.to_str(score))
        # Checkpoints
        if epoch % save_every == 0:
        	checkpoint = {
        		'epoch' : epoch,
        		'train_loss' : train_loss,
        		'val_loss' : val_loss,
        		'state_dict' : model.state_dict()
        	}
        	torch.save(
        		checkpoint,
        		'{}/{}-{}-{}-{}.pth'.format(
        			save_location, save_prefix,
        			epoch, train_loss, val_loss
        		)
        	)
        	print('Checkpoint saved')
    print(
        '\nTraining Done.\nTraining Mean Loss: {:6f}\nValidation Mean Loss: {:6f}'.format(
            sum(train_loss_history) / epochs,
            sum(val_loss_history) / epochs
        )
    )
    return train_loss_history, val_loss_history 


def main():

    print("checking device", "cuda:"+str(gpu_id))
    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    print("using DEVICE", device)

    train_images = sorted(glob('/local/DEEPLEARNING/camvid/train/*'))
    train_labels = sorted(glob('/local/DEEPLEARNING/camvid/trainannot/*'))
    val_images = sorted(glob('/local/DEEPLEARNING/camvid/val/*'))
    val_labels = sorted(glob('/local/DEEPLEARNING/camvid/valannot/*'))
    test_images = sorted(glob('/local/DEEPLEARNING/camvid/test/*'))
    test_labels = sorted(glob('/local/DEEPLEARNING/camvid/testannot/*'))
    batch_size = 10

    train_dataset = CamVidDataset(train_images, train_labels, 512, 512)
    print("length train", train_dataset.__len__())
    val_dataset = CamVidDataset(val_images, val_labels, 512, 512)
    print("length val", val_dataset.__len__())
    test_dataset = CamVidDataset(test_images, test_labels, 512, 512)
    print("length test", test_dataset.__len__())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    resnet50 = models.resnet50(pretrained=True)
    resnet50_backbone = models._utils.IntermediateLayerGetter(resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'})
    model = resvit.ResViT(pretrained_net=resnet50_backbone, num_class=12, dim=768, depth=1, heads=1, batch_size = batch_size, trans_img_size=(45,60))
    print("created resvit model")
    model.to(device)
    print("model put into ", device)
    class_weights = get_class_weights(train_loader, 12)
    criterion = CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )


    train_loss_history, val_loss_history = train(
    model, train_loader, val_loader,
    device, criterion, optimizer,
    len(train_images) // batch_size,
    len(val_images) // batch_size, 5,
    25, '/users/a/araujofj/camvid/checkpoints', 'resvit-model', epochs
    )

    print("calculating performance on test set")
    metrics = StreamSegMetrics(12)
    score = validate(model, test_loader, device, metrics)
    print(metrics.to_str(score))

    plt.figure(figsize=(20,10))
    plt.plot(train_loss_history)
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Loss function evolution")
    plt.savefig('loss_train.png')



if __name__ == '__main__':
    main()