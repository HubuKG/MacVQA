import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        """输入形状: (n_objects, 2048)"""
        attn_weights = self.attention(x)  # (n_objects, 1)
        return torch.sum(x * attn_weights, dim=0)  # (2048,)

class Autoencoder(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


def fuse_features(local_feat, global_feat):
    gate = np.tanh(local_feat + global_feat)
    return local_feat * gate + global_feat * (1 - gate)

def process_group(h5_file, group_name, global_feature_extractor, autoencoder):
    if "features" in h5_file[group_name]:
        features = np.array(h5_file[f"{group_name}/features"])
        boxes = np.array(h5_file[f"{group_name}/boxes"])
        img_h = np.array(h5_file[f"{group_name}/img_h"])
        img_w = np.array(h5_file[f"{group_name}/img_w"])
        obj_conf = np.array(h5_file[f"{group_name}/obj_conf"])
        obj_id = np.array(h5_file[f"{group_name}/obj_id"])

        enhanced_features = []
        for img_features in features:
            if img_features.ndim == 1:
                img_features = img_features.reshape(1, -1)
            
            img_tensor = torch.tensor(img_features, dtype=torch.float32)
            
            # 全局特征提取
            with torch.no_grad():
                global_feat = global_feature_extractor(img_tensor).numpy()
            
            # 特征融合
            fused_feat = fuse_features(img_features, global_feat)
            
            # 去噪处理（新增模型状态控制）
            autoencoder.eval()
            with torch.no_grad():
                denoised = autoencoder(
                    torch.tensor(fused_feat, dtype=torch.float32)
                ).detach().numpy()
            
            enhanced_features.append(denoised)

        return {
            "features": np.array(enhanced_features),
            "boxes": boxes,
            "img_h": img_h,
            "img_w": img_w,
            "obj_conf": obj_conf,
            "obj_id": obj_id
        }
    else:
        print(f"跳过无特征组: {group_name}")
        return None

def enhance_and_denoise_features(input_h5_path, output_h5_path):
    with h5py.File(input_h5_path, "r") as h5_file:
        first_group = next(iter(h5_file.keys()))
        input_dim = h5_file[f"{first_group}/features"].shape[1]
        
        assert input_dim == 2048, f"输入特征维度应为2048，实际为{input_dim}"
        
        # 初始化模型
        global_extractor = GlobalFeatureExtractor()
        autoencoder = Autoencoder()
        
        with h5py.File(output_h5_path, "w") as output_h5_file:
            for group_name in h5_file.keys():
                print(f"正在处理: {group_name}")
                processed_data = process_group(h5_file, group_name, global_extractor, autoencoder)
                
                if processed_data:
                    group = output_h5_file.create_group(group_name)
                    for key, value in processed_data.items():
                        group.create_dataset(key, data=value)
    
    print(f"处理完成！结果已保存至: {output_h5_path}")

if __name__ == "__main__":
    input_output_paths = {
        "/MacVQA/datasets/COCO/features/train2014_obj36.h5": "/root/autodl-tmp/VQACL/datasets/COCO/features/train2014_enhanced.h5",
        "/MacVQA/datasets/COCO/features/val2014_obj36.h5": "/root/autodl-tmp/VQACL/datasets/COCO/features/val2014_enhanced.h5",
        "/MacVQA/datasets/COCO/features/test2015_obj36.h5": "/root/autodl-tmp/VQACL/datasets/COCO/features/test2015_enhanced.h5",
    }
    
    for input_path, output_path in input_output_paths.items():
        if Path(output_path).exists():
            print(f"文件已存在: {output_path}")
            continue
            
        print(f"开始处理: {input_path}")
        enhance_and_denoise_features(input_path, output_path)
