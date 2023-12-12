import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import methods.util as util

UNKNOWN_LABEL = -1

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

cifar_mean = (0.5,0.5,0.5)
cifar_std = (0.25,0.25,0.25)

tiny_mean = (0.5,0.5,0.5)
tiny_std = (0.25,0.25,0.25)

svhn_mean = (0.5,0.5,0.5)
svhn_std = (0.25,0.25,0.25)

workers = 6
test_workers = 6
use_droplast = True
require_org_image = True
no_test_transform = False

DATA_PATH = 'data'
TINYIMAGENET_PATH = DATA_PATH + '/tiny-imagenet-200/'
LARGE_OOD_PATH = '/HOME/scz1838/run/largeoodds'
IMAGENET_PATH = '/data/public/imagenet2012'


class tinyimagenet_origin(Dataset):

    def __init__(self, _type, transform):
        if _type == 'train':
            self.ds = datasets.ImageFolder(f'{TINYIMAGENET_PATH}/train/', transform=transform)
            self.labels = [self.ds.samples[i][1] for i in range(len(self.ds))]
        elif _type == 'test':
            tmp_ds = datasets.ImageFolder(f'{TINYIMAGENET_PATH}/train/', transform=transform)
            cls2idx = tmp_ds.class_to_idx
            self.ds = datasets.ImageFolder(f'{TINYIMAGENET_PATH}/val/', transform=transform)
            with open(f'{TINYIMAGENET_PATH}/val/val_annotations.txt','r') as f:
                file2cls = {}
                for line in f.readlines():
                    line = line.strip().split('\t')
                    file2cls[line[0]] = line[1]
            self.labels = []
            for i in range(len(self.ds)):
                filename = self.ds.samples[i][0].split('/')[-1]
                self.labels.append(cls2idx[file2cls[filename]])
            # print("test labels",self.labels)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        return self.ds[idx][0],self.labels[idx]
    
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import random
class tinyimagenet_data(Dataset):

    def __init__(self, _type,transform):

        """
        Function:
            Given a dataset, construct its OpenCutmix-version dataset.

        Input:
            param1: tuple (imgs, targets),   param2: int (how many batches you want to generate, referring to your training batches.)

        Usage:
            # Build OpenCutmix dataset. It will take a few minutes.
            myOpenCutmix = OpenCutmix(trainset, batch_num)

            # Generate a batch of OpenCutmix samples. It will take a few seconds.
            myOpenCutmix.generate()

        """

        if _type=='train':
            transform=transforms.ToTensor()
        self.trainset = tinyimagenet_origin(_type,transform)
        self.negative_samples = []
        self.auxiliary_positive_samples = []
        self.original_labels = []
        self.trivial_masks = []

        self.negative_sample_num = 0
        self.batch_num = 10
        self.point = 0
        self.order = []

        self.grabcut()
        self.batchsize = int(self.negative_sample_num / self.batch_num)
        self.shuffle()

        audata,aulabel=self.generate()
        odata=[]
        olabel=[]
        for d,l in self.trainset:
            odata.append(d)
            olabel.append(l)
        odata=torch.stack(odata,0)
        olabel=torch.tensor(olabel)
        self.ds=torch.cat([odata,audata])
        self.labels=torch.cat([olabel,aulabel])


    def shuffle(self):
        self.order = torch.randperm(self.negative_sample_num)
        self.point = 0

    def grabcut(self):

        progress_bar = tqdm(self.trainset)
        for i, (images, labels) in enumerate(progress_bar):

            images = torch.unsqueeze(images, dim=0)
            labels = torch.tensor(labels)
            labels = torch.unsqueeze(labels, dim=0)

            # Get images, their foreground masks, and original labels using grabcut algorithm.
            # Bit 1 stands for foreground.
            selected_pic, fg_mask, label = grabcut(images, labels, 0.1, 0.8)

            if selected_pic is None:
                continue

            trivial_mask = fg_mask.clone()

            # Leverage fg_mask to generate some trivial masks.
            trivial_mask = torchvision.transforms.Resize(int(selected_pic.shape[2] / 2))(trivial_mask)
            trivial_mask = (trivial_mask != 0) * 1
            buff = torch.zeros(fg_mask.shape)
            buff[:, :, int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3),
            int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3)] = trivial_mask
            trivial_mask = buff

            # Initialize
            self.auxiliary_positive_samples.append(selected_pic)
            self.negative_samples.append(fg_mask)
            self.original_labels.append(label)
            self.trivial_masks.append(trivial_mask)

        self.negative_sample_num = len(self.auxiliary_positive_samples)
        print("finish grabcut!")

    # Get a batch of data
    def get_data(self):

        if self.point + self.batchsize > self.negative_sample_num:
            self.shuffle()

        selected_index = [self.order[k].item() for k in range(self.point, self.point + self.batchsize)]

        selected_pic = torch.cat([self.auxiliary_positive_samples[k] for k in selected_index], dim=0)
        selected_fg = torch.cat([self.negative_samples[k] for k in selected_index], dim=0)
        selected_labels = torch.cat([self.original_labels[k] for k in selected_index], dim=0)

        self.point = self.point + self.batchsize

        return selected_pic, selected_fg, selected_labels

    # Generate trivial masks
    def get_trivial_mask(self):

        # random order
        shuffle_index = torch.randperm(self.negative_sample_num)
        # random location
        random_x = random.randint(0, self.trivial_masks[0].shape[1])
        random_y = random.randint(0, self.trivial_masks[0].shape[1])

        selected_index = [shuffle_index[k].item() for k in range(self.batchsize)]
        tensor = torch.cat([self.trivial_masks[k] for k in selected_index], dim=0)
        buff = torch.zeros([tensor.shape[0], tensor.shape[1], tensor.shape[2] * 2, tensor.shape[3] * 2])
        buff[:, :, int(buff.shape[2] / 4):int(buff.shape[2] / 4 * 3),
        int(buff.shape[3] / 4):int(buff.shape[3] / 4 * 3)] = tensor
        return_mask = buff[:, :, random_x:random_x + int(buff.shape[2] / 2),
                      random_y:random_y + int(buff.shape[2] / 2)]

        return return_mask

    # Generate a batch of samples including negative samples and auxiliary_positive_samples
    # to prevent shortcuts.
    def generate(self):

        selected_pic, selected_fg, selected_labels = self.get_data()
        masks = self.get_trivial_mask()[:selected_pic.shape[0]]
        # Get filler for replacing
        filler = flip_for_openset(selected_pic)

        negative_samples = selected_pic * ((selected_fg == 0) * 1)*0.5 + filler * ((selected_fg == 1) * 1)*0.5
        # No overlap between foregrounds and trivial masks
        masks_judge = selected_pic * ((masks == 0) * 1)
        auxiliary_positive_samples = selected_pic * ((masks == 0) * 1)*0.5 + filler * ((masks == 1) * 1)*0.5
        selected_index = torch.sum((selected_fg * ((masks_judge == 0) * 1)), dim=[1, 2, 3]) == 0
        auxiliary_positive_samples = auxiliary_positive_samples[selected_index]

        # Negative labels
        negative_sample_labels = selected_labels.clone()
        negative_sample_labels[:] = 200
        # Maintaining labels
        auxiliary_positive_sample_labels = selected_labels.clone()
        auxiliary_positive_sample_labels = auxiliary_positive_sample_labels[selected_index]
        # Optional label switch. Depend on main.py
        # for index_ in range(len(auxiliary_positive_sample_labels)):
        #     auxiliary_positive_sample_labels[index_] = splits.index(int(auxiliary_positive_sample_labels[index_]))

        data = torch.cat([negative_samples, auxiliary_positive_samples], dim=0)
        targets = torch.cat([negative_sample_labels, auxiliary_positive_sample_labels], dim=0)

        return data, targets
    
    def __getitem__(self, index):
        return self.ds[index],self.labels[index]
    
    def __len__(self):
        return len(self.ds)

def grabcut(images, targets, lowerbound, upperbound):

    """
        Function:
            Given images, output their foreground masks using grabcut algorithm.

        Input:
            param3: float (0<param3<1, filter out tiny foreground masks),   param4: float (0<param4<1, filter out large foreground masks)

    """



    selected_pic = None
    foreground = None
    labels = None

    for k in range(len(images)):

        # try:

        image = images[k]

        image_test = transforms.ToPILImage()(image)

        img = cv2.cvtColor(np.asarray(image_test), cv2.COLOR_RGB2BGR)

        img = Image.fromarray(img)

        img = np.array(img)

        height, width, _ = img.shape

        x1 = 1
        x2 = width - 2
        y1 = 1
        y2 = height - 2

        rect = (x1, y1, x2, y2)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[:] = cv2.GC_PR_FGD
        mask[0, :] = cv2.GC_BGD
        mask[height - 1, :] = cv2.GC_BGD
        mask[:, 0] = cv2.GC_BGD
        mask[:, width - 1] = cv2.GC_BGD

        # Dummy placeholders
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        # Iteratively extract foreground object from background
        cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

        # Remove background from image
        fg_mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        fg_mask = torchvision.transforms.ToTensor()(fg_mask)



        # filter out irregular foregrounds
        if (torch.sum(fg_mask) > height * width * lowerbound) and (
                torch.sum(fg_mask) < height * width * upperbound):

            tensor_buff = torch.unsqueeze(image, dim=0)
            if selected_pic == None:
                selected_pic = tensor_buff
            else:
                selected_pic = torch.cat([selected_pic, tensor_buff], dim=0)

            tensor_buff = torch.unsqueeze(fg_mask, dim=0)
            if foreground == None:
                foreground = tensor_buff
            else:
                foreground = torch.cat([foreground, tensor_buff], dim=0)

            tensor_buff = torch.unsqueeze(targets[k], dim=0)
            if labels == None:
                labels = tensor_buff
            else:
                labels = torch.cat([labels, tensor_buff], dim=0)


    return selected_pic, foreground, labels

def flip_for_openset(data):


    image_h, image_w = data.shape[2:]

    data = torchvision.transforms.RandomHorizontalFlip()(data)
    data = torchvision.transforms.RandomVerticalFlip()(data)

    data2 = data.clone()

    data_buff = torchvision.transforms.functional.vflip(data2)

    data2[:,:,:int(image_h/2),:] = data[:,:,int(image_h/2):,:]
    data2[:,:,int(image_h/2):,:] = data_buff[:,:,:int(image_h/2),:]

    data_buff = torchvision.transforms.functional.hflip(data2)

    data2[:,:,:,int(image_h/2):] = data2[:,:,:,0:int(image_h/2)]
    data2[:, :, :, 0:int(image_h / 2)] = data_buff[:,:,:,int(image_h/2):]

    data2 = transforms.RandomRotation(360)(data2)


    return data2

class Imagenet1000(Dataset):

    lab_cvt = None

    def __init__(self,istrain, transform):

        set = "train" if istrain else "val"
        self.ds = datasets.ImageFolder(f'{IMAGENET_PATH}/{set}/', transform=transform)
        self.labels = [self.ds.samples[i][1] for i in range(len(self.ds))]
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        return self.ds[idx]

class LargeOODDataset(Dataset):

    def __init__(self,ds_name,transform) -> None:
        super().__init__()
        data_path = f'{LARGE_OOD_PATH}/{ds_name}/'
        self.ds = datasets.ImageFolder(data_path, transform=transform)
        self.labels = [-1] * len(self.ds)
    
    def __len__(self,):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]


class PartialDataset(Dataset):

    def __init__(self,knwon_ds,lab_keep = None,lab_cvt = None) -> None:
        super().__init__()
        self.known_ds = knwon_ds
        labels = knwon_ds.labels
        if lab_cvt is None:  # by default, identity mapping
            lab_cvt = [i for i in range(1999)]
        if lab_keep is None:  # by default, keep positive labels
            lab_keep = [x for x in lab_cvt if x > -1]
        keep = {x for x in lab_keep}
        self.sample_indexes = [i for i in range(len(knwon_ds)) if lab_cvt[labels[i]] in keep]
        self.labels = [lab_cvt[labels[i]] for i in range(len(knwon_ds)) if lab_cvt[labels[i]] in keep]
        self.labrefl = lab_cvt

    def __len__(self) -> int:
        return len(self.sample_indexes) 

    def __getitem__(self, index: int):
        inp,lb = self.known_ds[self.sample_indexes[index]]
        return inp,self.labrefl[lb],index

class UnionDataset(Dataset):

    def __init__(self,ds_list) -> None:
        super().__init__()
        self.dslist = ds_list
        self.totallen = sum([len(ds) for ds in ds_list])
        self.labels = []
        for x in ds_list:
            self.labels += x.labels
    
    def __len__(self) -> int:
        return self.totallen
    
    def __getitem__(self, index: int):
        orgindex = index
        for ds in self.dslist:
            if index < len(ds):
                a,b,c = ds[index]
                return a,b,orgindex
            index -= len(ds)
        return None


def gen_transform(mean,std,crop = False,toPIL = False,imgsize = 32,testmode = False):
    t = []
    if toPIL:
        t.append(transforms.ToPILImage())
    if not testmode:
        return transforms.Compose(t)
    if crop:
        if imgsize > 200:
            t += [transforms.Resize(256),transforms.CenterCrop(imgsize)]
        else:
            t.append(transforms.CenterCrop(imgsize))
    # print(t)
    return transforms.Compose(t + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def gen_cifar_transform(crop = False, toPIL = False,testmode = False):
    return gen_transform(cifar_mean,cifar_std,crop,toPIL=toPIL,imgsize=32,testmode = testmode)

def gen_tinyimagenet_transform(crop = False,testmode = False):
    return gen_transform(tiny_mean,tiny_std,crop,False,imgsize=64,testmode = testmode)

def gen_imagenet_transform(crop = False, testmode = False):
    return gen_transform(imagenet_mean,imagenet_std,crop,False,imgsize=224,testmode = testmode)

def gen_svhn_transform(crop = False,toPIL = False,testmode = False):
    return gen_transform(svhn_mean,svhn_std,crop,toPIL=toPIL,imgsize=32,testmode = testmode)


class myCifar10(Dataset):

    def __init__(self, _type,transform):

        """
        Function:
            Given a dataset, construct its OpenCutmix-version dataset.

        Input:
            param1: tuple (imgs, targets),   param2: int (how many batches you want to generate, referring to your training batches.)

        Usage:
            # Build OpenCutmix dataset. It will take a few minutes.
            myOpenCutmix = OpenCutmix(trainset, batch_num)

            # Generate a batch of OpenCutmix samples. It will take a few seconds.
            myOpenCutmix.generate()

        """

        if _type=='train':
            transform=transforms.ToTensor()
        self.trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
        self.negative_samples = []
        self.auxiliary_positive_samples = []
        self.original_labels = []
        self.trivial_masks = []

        self.negative_sample_num = 0
        self.batch_num = 10
        self.point = 0
        self.order = []

        self.grabcut()
        self.batchsize = int(self.negative_sample_num / self.batch_num)
        self.shuffle()

        audata,aulabel=self.generate()
        odata=[]
        olabel=[]
        for d,l in self.trainset:
            odata.append(d)
            olabel.append(l)
        odata=torch.stack(odata,0)
        olabel=torch.tensor(olabel)
        self.ds=torch.cat([odata,audata])
        self.labels=torch.cat([olabel,aulabel])
        self.targets=self.labels


    def shuffle(self):
        self.order = torch.randperm(self.negative_sample_num)
        self.point = 0

    def grabcut(self):

        progress_bar = tqdm(self.trainset)
        for i, (images, labels) in enumerate(progress_bar):

            images = torch.unsqueeze(images, dim=0)
            labels = torch.tensor(labels)
            labels = torch.unsqueeze(labels, dim=0)

            # Get images, their foreground masks, and original labels using grabcut algorithm.
            # Bit 1 stands for foreground.
            selected_pic, fg_mask, label = grabcut(images, labels, 0.1, 0.8)

            if selected_pic is None:
                continue

            trivial_mask = fg_mask.clone()

            # Leverage fg_mask to generate some trivial masks.
            trivial_mask = torchvision.transforms.Resize(int(selected_pic.shape[2] / 2))(trivial_mask)
            trivial_mask = (trivial_mask != 0) * 1
            buff = torch.zeros(fg_mask.shape)
            buff[:, :, int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3),
            int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3)] = trivial_mask
            trivial_mask = buff

            # Initialize
            self.auxiliary_positive_samples.append(selected_pic)
            self.negative_samples.append(fg_mask)
            self.original_labels.append(label)
            self.trivial_masks.append(trivial_mask)

        self.negative_sample_num = len(self.auxiliary_positive_samples)
        print("finish grabcut!")

    # Get a batch of data
    def get_data(self):

        if self.point + self.batchsize > self.negative_sample_num:
            self.shuffle()

        selected_index = [self.order[k].item() for k in range(self.point, self.point + self.batchsize)]

        selected_pic = torch.cat([self.auxiliary_positive_samples[k] for k in selected_index], dim=0)
        selected_fg = torch.cat([self.negative_samples[k] for k in selected_index], dim=0)
        selected_labels = torch.cat([self.original_labels[k] for k in selected_index], dim=0)

        self.point = self.point + self.batchsize

        return selected_pic, selected_fg, selected_labels

    # Generate trivial masks
    def get_trivial_mask(self):

        # random order
        shuffle_index = torch.randperm(self.negative_sample_num)
        # random location
        random_x = random.randint(0, self.trivial_masks[0].shape[1])
        random_y = random.randint(0, self.trivial_masks[0].shape[1])

        selected_index = [shuffle_index[k].item() for k in range(self.batchsize)]
        tensor = torch.cat([self.trivial_masks[k] for k in selected_index], dim=0)
        buff = torch.zeros([tensor.shape[0], tensor.shape[1], tensor.shape[2] * 2, tensor.shape[3] * 2])
        buff[:, :, int(buff.shape[2] / 4):int(buff.shape[2] / 4 * 3),
        int(buff.shape[3] / 4):int(buff.shape[3] / 4 * 3)] = tensor
        return_mask = buff[:, :, random_x:random_x + int(buff.shape[2] / 2),
                      random_y:random_y + int(buff.shape[2] / 2)]

        return return_mask

    # Generate a batch of samples including negative samples and auxiliary_positive_samples
    # to prevent shortcuts.
    def generate(self):

        selected_pic, selected_fg, selected_labels = self.get_data()
        masks = self.get_trivial_mask()[:selected_pic.shape[0]]
        # Get filler for replacing
        filler = flip_for_openset(selected_pic)

        negative_samples = selected_pic * ((selected_fg == 0) * 1)*0.5 + filler * ((selected_fg == 1) * 1)*0.5
        # No overlap between foregrounds and trivial masks
        masks_judge = selected_pic * ((masks == 0) * 1)
        auxiliary_positive_samples = selected_pic * ((masks == 0) * 1)*0.5 + filler * ((masks == 1) * 1)*0.5
        selected_index = torch.sum((selected_fg * ((masks_judge == 0) * 1)), dim=[1, 2, 3]) == 0
        auxiliary_positive_samples = auxiliary_positive_samples[selected_index]

        # Negative labels
        negative_sample_labels = selected_labels.clone()
        negative_sample_labels[:] = 10
        # Maintaining labels
        auxiliary_positive_sample_labels = selected_labels.clone()
        auxiliary_positive_sample_labels = auxiliary_positive_sample_labels[selected_index]
        # Optional label switch. Depend on main.py
        # for index_ in range(len(auxiliary_positive_sample_labels)):
        #     auxiliary_positive_sample_labels[index_] = splits.index(int(auxiliary_positive_sample_labels[index_]))

        data = torch.cat([negative_samples, auxiliary_positive_samples], dim=0)
        targets = torch.cat([negative_sample_labels, auxiliary_positive_sample_labels], dim=0)

        return data, targets
    
    def __getitem__(self, index):
        return self.ds[index],self.labels[index]
    
    def __len__(self):
        return len(self.ds)


def get_cifar10(settype):
    if settype == 'train':
        trans = gen_cifar_transform()
        ds = myCifar10(settype,trans)
    else:
        ds = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=gen_cifar_transform(testmode=True))
    ds.labels = ds.targets
    return ds

def get_cifar100(settype):
    if settype == 'train':
        trans = gen_cifar_transform()
        ds = torchvision.datasets.CIFAR100(root=DATA_PATH, train=True, download=True, transform=trans)
    else:
        ds =  torchvision.datasets.CIFAR100(root=DATA_PATH, train=False, download=True, transform=gen_cifar_transform(testmode=True))
    ds.labels = ds.targets
    return ds

class mySVHN(Dataset):

    def __init__(self, _type,transform):

        """
        Function:
            Given a dataset, construct its OpenCutmix-version dataset.

        Input:
            param1: tuple (imgs, targets),   param2: int (how many batches you want to generate, referring to your training batches.)

        Usage:
            # Build OpenCutmix dataset. It will take a few minutes.
            myOpenCutmix = OpenCutmix(trainset, batch_num)

            # Generate a batch of OpenCutmix samples. It will take a few seconds.
            myOpenCutmix.generate()

        """

        if _type=='train':
            transform=transforms.ToTensor()
        self.trainset = torchvision.datasets.SVHN(root=DATA_PATH, split='train', download=True, transform=transform)
        self.negative_samples = []
        self.auxiliary_positive_samples = []
        self.original_labels = []
        self.trivial_masks = []

        self.negative_sample_num = 0
        self.batch_num = 10
        self.point = 0
        self.order = []

        self.grabcut()
        self.batchsize = int(self.negative_sample_num / self.batch_num)
        self.shuffle()

        audata,aulabel=self.generate()
        odata=[]
        olabel=[]
        for d,l in self.trainset:
            odata.append(d)
            olabel.append(l)
        odata=torch.stack(odata,0)
        olabel=torch.tensor(olabel)
        self.ds=torch.cat([odata,audata])
        self.labels=torch.cat([olabel,aulabel])
        self.targets=self.labels

    def shuffle(self):
        self.order = torch.randperm(self.negative_sample_num)
        self.point = 0

    def grabcut(self):

        progress_bar = tqdm(self.trainset)
        for i, (images, labels) in enumerate(progress_bar):

            images = torch.unsqueeze(images, dim=0)
            labels = torch.tensor(labels)
            labels = torch.unsqueeze(labels, dim=0)

            # Get images, their foreground masks, and original labels using grabcut algorithm.
            # Bit 1 stands for foreground.
            selected_pic, fg_mask, label = grabcut(images, labels, 0.1, 0.8)

            if selected_pic is None:
                continue

            trivial_mask = fg_mask.clone()

            # Leverage fg_mask to generate some trivial masks.
            trivial_mask = torchvision.transforms.Resize(int(selected_pic.shape[2] / 2))(trivial_mask)
            trivial_mask = (trivial_mask != 0) * 1
            buff = torch.zeros(fg_mask.shape)
            buff[:, :, int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3),
            int(selected_pic.shape[2] / 4):int(selected_pic.shape[2] / 4 * 3)] = trivial_mask
            trivial_mask = buff

            # Initialize
            self.auxiliary_positive_samples.append(selected_pic)
            self.negative_samples.append(fg_mask)
            self.original_labels.append(label)
            self.trivial_masks.append(trivial_mask)

        self.negative_sample_num = len(self.auxiliary_positive_samples)
        print("finish grabcut!")

    # Get a batch of data
    def get_data(self):

        if self.point + self.batchsize > self.negative_sample_num:
            self.shuffle()

        selected_index = [self.order[k].item() for k in range(self.point, self.point + self.batchsize)]

        selected_pic = torch.cat([self.auxiliary_positive_samples[k] for k in selected_index], dim=0)
        selected_fg = torch.cat([self.negative_samples[k] for k in selected_index], dim=0)
        selected_labels = torch.cat([self.original_labels[k] for k in selected_index], dim=0)

        self.point = self.point + self.batchsize

        return selected_pic, selected_fg, selected_labels

    # Generate trivial masks
    def get_trivial_mask(self):

        # random order
        shuffle_index = torch.randperm(self.negative_sample_num)
        # random location
        random_x = random.randint(0, self.trivial_masks[0].shape[1])
        random_y = random.randint(0, self.trivial_masks[0].shape[1])

        selected_index = [shuffle_index[k].item() for k in range(self.batchsize)]
        tensor = torch.cat([self.trivial_masks[k] for k in selected_index], dim=0)
        buff = torch.zeros([tensor.shape[0], tensor.shape[1], tensor.shape[2] * 2, tensor.shape[3] * 2])
        buff[:, :, int(buff.shape[2] / 4):int(buff.shape[2] / 4 * 3),
        int(buff.shape[3] / 4):int(buff.shape[3] / 4 * 3)] = tensor
        return_mask = buff[:, :, random_x:random_x + int(buff.shape[2] / 2),
                      random_y:random_y + int(buff.shape[2] / 2)]

        return return_mask

    # Generate a batch of samples including negative samples and auxiliary_positive_samples
    # to prevent shortcuts.
    def generate(self):

        selected_pic, selected_fg, selected_labels = self.get_data()
        masks = self.get_trivial_mask()[:selected_pic.shape[0]]
        # Get filler for replacing
        filler = flip_for_openset(selected_pic)

        negative_samples = selected_pic * ((selected_fg == 0) * 1)*0.5 + filler * ((selected_fg == 1) * 1)*0.5
        # No overlap between foregrounds and trivial masks
        masks_judge = selected_pic * ((masks == 0) * 1)
        auxiliary_positive_samples = selected_pic * ((masks == 0) * 1)*0.5 + filler * ((masks == 1) * 1)*0.5
        selected_index = torch.sum((selected_fg * ((masks_judge == 0) * 1)), dim=[1, 2, 3]) == 0
        auxiliary_positive_samples = auxiliary_positive_samples[selected_index]

        # Negative labels
        negative_sample_labels = selected_labels.clone()
        negative_sample_labels[:] = 10
        # Maintaining labels
        auxiliary_positive_sample_labels = selected_labels.clone()
        auxiliary_positive_sample_labels = auxiliary_positive_sample_labels[selected_index]
        # Optional label switch. Depend on main.py
        # for index_ in range(len(auxiliary_positive_sample_labels)):
        #     auxiliary_positive_sample_labels[index_] = splits.index(int(auxiliary_positive_sample_labels[index_]))

        data = torch.cat([negative_samples, auxiliary_positive_samples], dim=0)
        targets = torch.cat([negative_sample_labels, auxiliary_positive_sample_labels], dim=0)

        return data, targets
    
    def __getitem__(self, index):
        return self.ds[index],self.labels[index]
    
    def __len__(self):
        return len(self.ds)

def get_svhn(settype):
    if settype == 'train':
        trans = gen_svhn_transform()
        ds = mySVHN(settype,trans)
    else :
        ds = torchvision.datasets.SVHN(root=DATA_PATH, split='test', download=True, transform=gen_svhn_transform(testmode=True))
    return ds

def get_tinyimagenet(settype):
    if settype == 'train':
        trans = gen_tinyimagenet_transform()
        ds = tinyimagenet_data('train',trans)
    else:
        ds = tinyimagenet_origin('test',gen_tinyimagenet_transform(testmode=True))
    return ds

def get_imagenet1000(settype):
    if settype == 'train':
        trans = gen_imagenet_transform()
        ds = Imagenet1000(True,trans)
    else:
        ds = Imagenet1000(False,gen_imagenet_transform(crop = True, testmode=True))
    return ds

def get_ood_inaturalist(settype):
    if settype == 'train':
        raise Exception("OOD iNaturalist cannot be used as train set.")
    else:
        return LargeOODDataset('iNaturalist',gen_imagenet_transform(crop = True, testmode=True))

ds_dict = {
    "cifarova" : get_cifar10,
    "cifar10" : get_cifar10,
    "cifar100" : get_cifar100,
    "svhn" : get_svhn,
    "tinyimagenet" : get_tinyimagenet,
    "imagenet" : get_imagenet1000,
    'oodinaturalist' : get_ood_inaturalist,
}

cache_base_ds = {

}

def get_ds_with_name(settype,ds_name):
    global cache_base_ds
    key = str(settype) + ds_name
    if key not in cache_base_ds.keys():
        cache_base_ds[key] = ds_dict[ds_name](settype)
    return cache_base_ds[key]

def get_partialds_with_name(settype,ds_name,label_cvt,label_keep):
    ds = get_ds_with_name(settype,ds_name)
    return PartialDataset(ds,label_keep,label_cvt)
    
# setting list [[ds_name, sample partition list, label convertion list],...]
def get_combined_dataset(settype,setting_list):
    ds_list = []
    for setting in setting_list:
        ds = get_partialds_with_name(settype,setting['dataset'],setting['convert_class'],setting['keep_class'])
        if ds.__len__() > 0:
            ds_list.append(ds)
    return UnionDataset(ds_list) if len(ds_list) > 0 else None

def get_combined_dataloaders(args,settings):
    istrain_mode = True
    print("Load with train mode :",istrain_mode)
    train_labeled = get_combined_dataset('train',settings['train'])
    test = get_combined_dataset('test',settings['test'])
    return torch.utils.data.DataLoader(train_labeled, batch_size=args.bs, shuffle=istrain_mode, num_workers=workers,pin_memory=True,drop_last = use_droplast) if train_labeled is not None else None,\
            torch.utils.data.DataLoader(test, batch_size=args.bs, shuffle=False, num_workers=test_workers,pin_memory=args.gpu != 'cpu') if test is not None else None

ds_classnum_dict = {
    'cifar10' : 7,
    'svhn' : 7,
    'tinyimagenet' : 21,
    "imagenet" : 1000,
}

imgsize_dict = {
    'cifar10' : 32,
    'svhn' : 32,
    'tinyimagenet' : 64,
    "imagenet" : 224,
}

def load_partitioned_dataset(args,ds):
    with open(ds,'r') as f:
        settings = json.load(f)
    util.img_size = imgsize_dict[settings['name']]
    a,b = get_combined_dataloaders(args,settings)
    return a,b,ds_classnum_dict[settings['name']]

