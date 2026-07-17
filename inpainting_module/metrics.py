import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F

import lpips
import torch_fidelity


def psnr(img1, img2):
    """img1, img2: torch tensor [C,H,W], 范围[0,1]"""
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse)


def load_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in exts])
    return files


def main(folderA, folderB):
    # ---------------------
    # 1. 计算 FID
    # ---------------------
    fid_result = torch_fidelity.calculate_metrics(
        input1=folderA,
        input2=folderB,
        cuda=torch.cuda.is_available(),
        fid=True, isc=False, kid=False, verbose=False
    )
    print("FID:", fid_result["frechet_inception_distance"])

    # ---------------------
    # 2. 计算 PSNR / LPIPS
    # ---------------------
    filesA = load_images(folderA)
    filesB = load_images(folderB)
    if len(filesA) != len(filesB):
        print("两个文件夹图片数量不一致，按较小数量对齐。")
    n = min(len(filesA), len(filesB))

    to_tensor = transforms.ToTensor()
    lpips_fn = lpips.LPIPS(net="vgg")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips_fn.to(device)

    psnr_list, lpips_list = [], []
    for i in range(n):
        imgA = to_tensor(Image.open(filesA[i]).convert("RGB")).unsqueeze(0).to(device)
        imgB = to_tensor(Image.open(filesB[i]).convert("RGB")).unsqueeze(0).to(device)

        # resize 保证相同大小
        if imgA.shape != imgB.shape:
            h = min(imgA.shape[2], imgB.shape[2])
            w = min(imgA.shape[3], imgB.shape[3])
            imgA = F.interpolate(imgA, size=(h, w), mode="bilinear", align_corners=False)
            imgB = F.interpolate(imgB, size=(h, w), mode="bilinear", align_corners=False)

        # PSNR
        psnr_val = psnr(imgA.squeeze(0), imgB.squeeze(0)).item()
        psnr_list.append(psnr_val)

        # LPIPS 需要 [-1,1]
        lp = lpips_fn(imgA * 2 - 1, imgB * 2 - 1).item()
        lpips_list.append(lp)

        print(f"{os.path.basename(filesA[i])} vs {os.path.basename(filesB[i])}: "
              f"PSNR={psnr_val:.4f} dB, LPIPS={lp:.6f}")

    print("\n平均 PSNR:", np.mean(psnr_list))
    print("平均 LPIPS:", np.mean(lpips_list))


if __name__ == "__main__":
    # 使用方法：修改下面两行路径
    folderA = "metric/images"
    folderB = "metric/ours"
    main(folderA, folderB)