import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir, getsize, split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import cv2

def cointoss(p):
    return random.random() < p

def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth):
    if bit_depth == 8:
        stream.seek(iFrame*1.5*width*height)
        Y = np.fromfile(stream, dtype=np.uint8, count=width*height).reshape((height, width))
        
        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))
        V = np.fromfile(stream, dtype=np.uint8, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))

    else:
        stream.seek(iFrame*3*width*height)
        Y = np.fromfile(stream, dtype=np.uint16, count=width*height).reshape((height, width))
                
        U = np.fromfile(stream, dtype=np.uint16, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))
        V = np.fromfile(stream, dtype=np.uint16, count=(width//2)*(height//2)).\
                                reshape((height//2, width//2))

    
    yuv = np.empty((height*3//2, width), dtype=np.uint8)
    yuv[0:height,:] = Y

    yuv[height:height+height//4,:] = U.reshape(-1, width)
    yuv[height+height//4:,:] = V.reshape(-1, width)

    #convert to rgb
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    # bgr = cv2.resize(bgr, (int(width/4),int(height/4)), interpolation = cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # (270,480,3)

    return rgb


class DBreader_Vimeo90k(Dataset):
    def __init__(self, db_dir, random_crop=None, resize=None, augment_s=True, augment_t=True, train=True):
        seq_dir = join(db_dir, 'sequences')
        self.random_crop = random_crop
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = []
        if resize is not None:
            transform_list += [transforms.Resize(resize)]

        transform_list += [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        if train:
            seq_list_txt = join(db_dir, 'tri_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'tri_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]
        self.file_len = len(self.seq_path_list)

    def __getitem__(self, index):
        rawFrame0 = Image.open(join(self.seq_path_list[index],  "im1.png"))
        rawFrame1 = Image.open(join(self.seq_path_list[index],  "im2.png"))
        rawFrame2 = Image.open(join(self.seq_path_list[index],  "im3.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
        
        if cointoss(0.5):
            brightness_factor = random.uniform(0.95, 1.05)
            rawFrame0 = TF.adjust_brightness(rawFrame0, brightness_factor)
            rawFrame1 = TF.adjust_brightness(rawFrame1, brightness_factor)
            rawFrame2 = TF.adjust_brightness(rawFrame2, brightness_factor)
            contrast_factor = random.uniform(0.95, 1.05)
            rawFrame0 = TF.adjust_contrast(rawFrame0, contrast_factor)
            rawFrame1 = TF.adjust_contrast(rawFrame1, contrast_factor)
            rawFrame2 = TF.adjust_contrast(rawFrame2, contrast_factor)
            saturation_factor = random.uniform(0.95, 1.05)
            rawFrame0 = TF.adjust_saturation(rawFrame0, saturation_factor)
            rawFrame1 = TF.adjust_saturation(rawFrame1, saturation_factor)
            rawFrame2 = TF.adjust_saturation(rawFrame2, saturation_factor)
            hue_factor = random.uniform(-0.05, 0.05)
            rawFrame0 = TF.adjust_hue(rawFrame0, hue_factor)
            rawFrame1 = TF.adjust_hue(rawFrame1, hue_factor)
            rawFrame2 = TF.adjust_hue(rawFrame2, hue_factor)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return self.file_len
    

class BVIDVC(Dataset):
    def __init__(self, db_dir, res, crop_sz=(256,256), augment_s=True, augment_t=True):
        assert res in ['2k', '1080', '960', '480']

        db_dir = join(db_dir, 'Videos')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        transform_list = [transforms.ToTensor()]

        self.transform = transforms.Compose(transform_list)

        prefix = {'2k': 'A', '1080': 'B', '960': 'C', '480': 'D'}
        self.seq_path_list = [join(db_dir, f) for f in listdir(db_dir) \
                              if f.startswith(prefix[res]) and f.endswith('.yuv')]

    def __getitem__(self, index):
        # first randomly sample a triplet
        stream = open(self.seq_path_list[index], 'r')
        _, fname = split(self.seq_path_list[index])
        width, height = [int(i) for i in fname.split('_')[1].split('x')]
        file_size = getsize(self.seq_path_list[index])
        num_frames = file_size // (width*height*3)
        frame_idx = random.randint(1, num_frames-2)

        rawFrame0 = Image.fromarray(read_frame_yuv2rgb(stream, width, height, frame_idx-1, 10))
        rawFrame1 = Image.fromarray(read_frame_yuv2rgb(stream, width, height, frame_idx, 10))
        rawFrame2 = Image.fromarray(read_frame_yuv2rgb(stream, width, height, frame_idx+1, 10))
        stream.close()

        if self.crop_sz is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.crop_sz)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if cointoss(0.5):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if cointoss(0.5):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
        
        if cointoss(0.5):
            brightness_factor = random.uniform(0.95, 1.05)
            rawFrame0 = TF.adjust_brightness(rawFrame0, brightness_factor)
            rawFrame1 = TF.adjust_brightness(rawFrame1, brightness_factor)
            rawFrame2 = TF.adjust_brightness(rawFrame2, brightness_factor)
            contrast_factor = random.uniform(0.95, 1.05)
            rawFrame0 = TF.adjust_contrast(rawFrame0, contrast_factor)
            rawFrame1 = TF.adjust_contrast(rawFrame1, contrast_factor)
            rawFrame2 = TF.adjust_contrast(rawFrame2, contrast_factor)
            saturation_factor = random.uniform(0.95, 1.05)
            rawFrame0 = TF.adjust_saturation(rawFrame0, saturation_factor)
            rawFrame1 = TF.adjust_saturation(rawFrame1, saturation_factor)
            rawFrame2 = TF.adjust_saturation(rawFrame2, saturation_factor)
            hue_factor = random.uniform(-0.05, 0.05)
            rawFrame0 = TF.adjust_hue(rawFrame0, hue_factor)
            rawFrame1 = TF.adjust_hue(rawFrame1, hue_factor)
            rawFrame2 = TF.adjust_hue(rawFrame2, hue_factor)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if cointoss(0.5):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2

    def __len__(self):
        return len(self.seq_path_list)


class Sampler(Dataset):
    def __init__(self, datasets, p_datasets=None, iter=False, samples_per_epoch=1000):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch