import torch

from util import *
from dataloader import CSVDataset
from torch.utils.data import Dataset, DataLoader
import sys


def main(train_file, cls_file):
	dataset = CSVDataset(train_file = train_file, class_list=cls_file)
	mean, std = get_mean_and_std(dataset)
	print('mean, std:', mean, std)


if __name__ == '__main__':
	train_file = sys.argv[1]
	cls_file = sys.argv[2]
	main(train_file, cls_file)

