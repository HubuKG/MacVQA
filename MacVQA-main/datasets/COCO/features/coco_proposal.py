# coding=utf-8

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
from diffusers import StableDiffusionPipeline  # 引入扩散模型
from PIL import Image
import torchvision.transforms as transforms
import h5py
import numpy as np


class COCODataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)

        # self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }


def generate_images(prompt, num_images=5):
    """使用扩散模型生成图像"""
    print(f"Generating {num_images} images with prompt: {prompt}")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda")
    generated_images = pipe(prompt, num_inference_steps=50).images[:num_images]
    return generated_images


def extract_features_with_augmentation(image_path, save_path, group_name, predictor=None):
    """提取原始特征和增强特征"""
    print(f"Extracting features for: {image_path}")
    image = cv2.imread(image_path)

    # 模拟原始特征提取（此处可以替换为真实的特征提取逻辑）
    original_features = np.random.rand(NUM_OBJECTS, DIM)  # 假设每张图有 NUM_OBJECTS 个对象，每个对象 DIM 维特征

    # 对特征添加噪声增强
    augmented_features = original_features + np.random.normal(0, 0.1, original_features.shape)

    # 保存特征到 h5 文件
    with h5py.File(save_path, "a") as f:  # 使用 "a" 模式追加数据
        grp = f.create_group(group_name)
        grp['original_features'] = original_features
        grp['augmented_features'] = augmented_features
    print(f"Features saved under group: {group_name}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--cocoroot', type=str, default='/ssd-playpen/home/jmincho/workspace/datasets/COCO/')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--prompt', type=str, default='A beautiful landscape with mountains and a river',
                        help='扩散模型的文本提示')
    parser.add_argument('--num_generated_images', type=int, default=5,
                        help='每组生成的图像数量')

    args = parser.parse_args()

    SPLIT2DIR = {
        'train': 'train2014',
        'valid': 'val2014',
        'test': 'test2015',
    }

    coco_dir = Path(args.cocoroot).resolve()
    coco_img_dir = coco_dir.joinpath('images')
    coco_img_split_dir = coco_img_dir.joinpath(SPLIT2DIR[args.split])

    dataset_name = 'COCO'

    out_dir = coco_dir.joinpath('features')
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', coco_img_split_dir)
    print('# Images:', len(list(coco_img_split_dir.iterdir())))

    dataset = COCODataset(coco_img_split_dir)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    # 原始特征提取
    extract(output_fname, dataloader, desc)

    # 扩散模型生成图像并提取特征
    generated_images = generate_images(args.prompt, args.num_generated_images)
    for i, img in enumerate(generated_images):
        # 保存生成图像的特征
        img_save_path = out_dir.joinpath(f'generated_image_{i}.jpg')  # 临时保存图像
        img.save(img_save_path)
        extract_features_with_augmentation(str(img_save_path), output_fname, group_name=f"generated_image_{i}")
