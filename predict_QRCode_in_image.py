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

import skimage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, \
    Normalizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROI_mean = (0.1146, 0.1147, 0.1148)
ROI_std = (0.1089, 0.1090, 0.1090)
QRCode_mean = (0.2405, 0.2416, 0.2427)
QRCode_std = (0.2194, 0.2208, 0.2223)


output_csv_fn = 'output.csv'
output_file = open(output_csv_fn, 'w')

not_processed_fn = 'not_processed.csv'
not_processed_file = open(not_processed_fn, 'w')

debug = True

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple visualizing script for visualize a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--ROI_model', help='Path to ROI model (.pt) file.')
    parser.add_argument('--QRCode_model', help="path to QRcode model(.pt) file")

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(ROI_mean, ROI_std), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=None, sampler=None)

    ROI_net = torch.load(parser.ROI_model)
    QRCode_net = torch.load(parser.QRCode_model)

    use_gpu = True

    if use_gpu:
        ROI_net = ROI_net.cuda()
        QRCode_net = QRCode_net.cuda(0)

    ROI_net.eval()
    QRCode_net.eval()

    unnormalize = UnNormalizer(ROI_mean, ROI_std)

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = ROI_net(data['img'].cuda().float())
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

            if idxs[0].shape[0] == 1:
                origin_img = cv2.imread(fn)
                ph, pw, _ = img.shape
                ret = convert_predict_to_origin_bbox(origin_img, pw, ph, x1, y1, x2, y2)
                if ret is None:
                    print("ERROR: convert predicted origin bbox error")
                    continue

                x1p, y1p, x2p, y2p = ret
                print("ROI predicted:", x1p, y1p, x2p, y2p)
                output_file.write(fn+','+str(x1p)+','+str(y1p)+','+str(x2p)+','+str(y2p)+',ROI\n')
                print("!!!! FN {} saved!!!".format(fn))
                ROI = origin_img[y1p:y2p, x1p:x2p]
                cv2.rectangle(origin_img, (x1p, y1p), (x2p, y2p), color=(0,0,255), thickness=8)
                #import pdb
                #pdb.set_trace()
                ROI = ROI.astype(np.float32)/255.0
                # normalize it
                ROI_normalized = (ROI-QRCode_mean)/QRCode_std
                #resize it
                rows, cols, cns = ROI_normalized.shape
                smallest_side = min(rows, cols)
                #rescale the image so the smallest side is min_side
                min_side = 600.0
                max_side = 900.0
                scale = min_side/smallest_side
                #check if the largest side is now greater than max_side, which can happen
                # when images have a large aspect ratio
                largest_side = max(rows, cols)
                if largest_side * scale > 900:
                    scale = max_side / largest_side

                # resize the image with the computed scale
                ROI_scale = skimage.transform.resize(ROI_normalized, (int(round(rows*scale)), int(round((cols*scale)))))
                rows, cols, cns = ROI_scale.shape

                pad_w = 32 - rows%32
                pad_h = 32 - cols%32

                ROI_padded = np.zeros((rows+pad_w, cols+pad_h, cns)).astype(np.float32)
                ROI_padded[:rows, :cols, :] = ROI_scale.astype(np.float32)
                x = torch.from_numpy(ROI_padded)
                print('x.shape:', x.shape)
                x = torch.unsqueeze(x, dim=0)
                print('x.shape after unsqueeze:', x.shape)
                x = x.permute(0,3,1,2)
                print('x.shape after permute:', x.shape)

                scores, classification, transformed_anchors = QRCode_net(x.cuda().float())
                print('scores:', scores)
                print('classification;', classification)
                print('transformed_anchors:', transformed_anchors)
                idxs = np.where(scores.cpu()>0.5)
                predict_height, predict_width, _ = ROI_padded.shape

                for j in range(idxs[0].shape[0]):
                    bbox = transformed_anchors[idxs[0][j],:]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    print("!!QRCode predicted bbox inside ROI:", x1, y1, x2, y2)

                    ret = convert_predict_to_origin_bbox(ROI, predict_width, predict_height, x1, y1, x2, y2)
                    if ret is None:
                        continue

                    qrcode_x1, qrcode_y1, qrcode_x2, qrcode_y2 = ret
                    print('qrcode(bbox):', qrcode_x1, qrcode_y1, qrcode_x2, qrcode_y2)

                    qrcode_img_x1 = x1p + qrcode_x1
                    qrcode_img_y1 = y1p + qrcode_y1
                    qrcode_img_x2 = x1p + qrcode_x2
                    qrcode_img_y2 = y1p + qrcode_y2
                    print('!!!QRCode in image:', qrcode_img_x1, qrcode_img_y1, qrcode_img_x2, qrcode_img_y2)
                    cv2.rectangle(origin_img, (qrcode_img_x1, qrcode_img_y1), (qrcode_img_x2, qrcode_img_y2), color=(255,0,0), thickness=8)
                    cv2.imwrite('origin_img_qrcode.png', origin_img)
                    resized = cv2.resize(origin_img, (800,600))
                    cv2.imshow('result', resized)
            else:
                not_processed_file.write(fn+",,,,,\n")

            if debug:
                cv2.imshow('img', img)
                cv2.setWindowTitle('img', fn)
                key = cv2.waitKey(0)
                if 'q'==chr(key & 255):
                    exit(0)

    output_file.close()
    not_processed_file.close()

def convert_predict_to_origin_bbox(origin_img, predict_width, predict_height, x1, y1, x2, y2):
    rows, cols, cns = origin_img.shape
    pad_w = 32-rows%32
    pad_h = 32-cols%32
    pad_height = rows+pad_h

    predict_aspect_ratio=predict_width*1.0/predict_height
    target_width = int(pad_height*predict_aspect_ratio)

    try:
        new_image = np.zeros((pad_height, target_width, cns))
        new_image[:rows, :cols,:] = origin_img

        img_scale = float(target_width)/predict_width
        x1 = int(x1*img_scale)
        y1 = int(y1*img_scale)
        x2 = int(x2*img_scale)
        y2 = int(y2*img_scale)

        return x1, y1, x2, y2
    except:
        return None


if __name__ == '__main__':
    main()
