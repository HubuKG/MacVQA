# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
from tqdm import tqdm
import numpy as np
import h5py
import argparse
from diffusers import StableDiffusionPipeline  # 引入扩散模型
from PIL import Image
import torchvision.transforms as transforms

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in tqdm(enumerate(reader), ncols=150):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(
                    base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." %
          (len(data), fname, elapsed_time))
    return data

def generate_images(prompt, num_images=5):
    """使用扩散模型生成图像"""
    print(f"Generating {num_images} images with prompt: {prompt}")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda")
    generated_images = pipe(prompt, num_inference_steps=50).images[:num_images]
    return generated_images

def extract_features(images, save_path, group_name):
    """提取生成图像的特征并保存到 h5 文件"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    with h5py.File(save_path, "a") as f:  # 使用 "a" 模式追加数据
        grp = f.create_group(group_name)
        for i, img in enumerate(images):
            img_tensor = transform(img).unsqueeze(0)
            # 模拟特征提取（此处可以替换为真实的特征提取逻辑）
            features = np.random.rand(36, 2048)  # 假设每张图有 36 个对象，每个对象 2048 维特征
            grp[f"generated_features_{i}"] = features
    print(f"Generated features saved under group: {group_name}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str,
                        default='val2014_obj36.tsv')
    parser.add_argument('--h5_path', type=str,
                        default='val2014_obj36.h5')
    parser.add_argument('--prompt', type=str,
                        default='A beautiful landscape with mountains and a river',
                        help='扩散模型的文本提示')
    parser.add_argument('--num_generated_images', type=int, default=5,
                        help='每组生成的图像数量')

    args = parser.parse_args()
    dim = 2048

    print('Load ', args.tsv_path)
    data = load_obj_tsv(args.tsv_path)
    print('# data:', len(data))

    output_fname = args.h5_path
    print('features will be saved at', output_fname)

    with h5py.File(output_fname, 'w') as f:
        for i, datum in tqdm(enumerate(data),
                            ncols=150,):

            img_id = datum['img_id']

            num_boxes = datum['num_boxes']

            grp = f.create_group(img_id)
            grp['features'] = datum['features'].reshape(num_boxes, 2048)
            grp['obj_id'] = datum['objects_id']
            grp['obj_conf'] = datum['objects_conf']
            grp['attr_id'] = datum['attrs_id']
            grp['attr_conf'] = datum['attrs_conf']
            grp['boxes'] = datum['boxes']
            grp['img_w'] = datum['img_w']
            grp['img_h'] = datum['img_h']

    # 扩散模型生成图像并提取特征
    generated_images = generate_images(args.prompt, args.num_generated_images)
    extract_features(generated_images, args.h5_path, group_name="generated_images")
