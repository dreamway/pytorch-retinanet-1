import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2
import model

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, \
    Normalizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean = (0.1146, 0.1147, 0.1148)
std = (0.1089, 0.1090, 0.1090)

output_csv_fn = 'output.csv'
output_file = open(output_csv_fn, 'w')


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple visualizing script for visualize a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(mean, std), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    #sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=None, sampler=None)

    retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        retinanet = retinanet.cuda()

    retinanet.eval()

    unnormalize = UnNormalizer(mean, std)

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            # if batch_size = 1, and batch_sampler, sampler is None, then no_shuffle, will use sequential index, then the get_image_name is OK.
            # otherwise, it will failed.
            fn = dataset_val.get_image_name(idx)
            print('fn of image:', fn)
            idxs = np.where(scores.cpu() > 0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            print("image shape when drawcaption:", img.shape)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if 'q'==chr(key & 255):
                exit(0)
            elif 's'==chr(key & 255):
                origin_img = cv2.imread(fn)
                origin_height, origin_width, _ = origin_img.shape
                predict_height, predict_width, _ = img.shape

                x1p, y1p, x2p, y2p = convert_predict_to_origin_bbox(origin_img, img, x1, y1, x2, y2)
                print('x1p,y1p,x2p,y2p:', x1p,y1p, x2p, y2p)
                output_file.write(fn+','+str(x1p)+','+str(y1p)+','+str(x2p)+','+str(y2p)+','+'ROI\n')
                print("!!!!!!!!!!! FN: {} SAVED!!!!!".format(fn))
            else:
                pass
    output_file.close()



def convert_predict_to_origin_bbox(origin_img, img, x1, y1, x2, y2):
    rows, cols, cns = origin_img.shape
    pad_w = 32-rows%32
    pad_h = 32-cols%32
    pad_height = rows+pad_h
    predict_height, predict_width, _ = img.shape

    predict_aspect_ratio=predict_width*1.0/predict_height
    target_width = int(pad_height*predict_aspect_ratio)
    new_image = np.zeros((pad_height, target_width, cns))
    new_image[:rows, :cols,:] = origin_img

    img_scale = float(target_width)/predict_width
    x1 = int(x1*img_scale)
    y1 = int(y1*img_scale)
    x2 = int(x2*img_scale)
    y2 = int(y2*img_scale)

    return x1, y1, x2, y2


if __name__ == '__main__':
    main()
