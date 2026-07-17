from PIL import Image


def apply_mask(base_image_path, mask_image_path, output_path):
    # 打开底图和 mask 图像
    base_img = Image.open(base_image_path).convert("RGB")
    mask_img = Image.open(mask_image_path).convert("RGB")

    # 确保两张图尺寸一致
    if base_img.size != mask_img.size:
        mask_img = mask_img.resize(base_img.size)

    # 获取像素数据
    base_pixels = base_img.load()
    mask_pixels = mask_img.load()

    width, height = base_img.size

    # 将 mask 中的黑色像素覆盖到底图上
    for y in range(height):
        for x in range(width):
            r, g, b = mask_pixels[x, y]
            if r < 10 and g < 10 and b < 10:  # 近似判断为黑色
                base_pixels[x, y] = (0, 0, 0)

    # 保存输出图像
    base_img.save(output_path)
    print(f"已保存结果图像: {output_path}")


# 示例使用
apply_mask("test_sets/CelebA-HQ/images/6910.png", "test_sets/CelebA-HQ/masks/002352.png", "output.jpg")