import os
import time
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pdf2image import convert_from_path


def convert_pdf_to_images(pdf_path, output_dir):
    """将 PDF 转换为图像"""
    images = convert_from_path(pdf_path, first_page=2, last_page=3)  # 只处理前两页
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def compare_and_mask_images(
    img1, img2, window_size=150, stride=80, similarity_threshold=0.95
):
    # 检查图片是否成功加载
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # 检查图片尺寸是否一致
    if img1.shape != img2.shape:
        print("Error: Images have different dimensions.")
        return

    # 获取图片尺寸
    height, width, _ = img1.shape

    # 创建掩码（初始化为全零）
    mask = np.zeros((height, width), dtype=np.uint8)

    # 滑窗逐行扫描
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # 提取滑窗区域
            window1 = img1[y : y + window_size, x : x + window_size]
            window2 = img2[y : y + window_size, x : x + window_size]

            # 转换为灰度图以计算 SSIM
            window1_gray = cv2.cvtColor(window1, cv2.COLOR_BGR2GRAY)
            window2_gray = cv2.cvtColor(window2, cv2.COLOR_BGR2GRAY)

            # 计算 SSIM 相似度
            similarity, _ = ssim(window1_gray, window2_gray, full=True)

            # 如果相似度超过阈值，标记为重叠区域
            if similarity > similarity_threshold:
                mask[y : y + window_size, x : x + window_size] = 255
    return mask

def main(pdf_path, output_dir):
    """主函数"""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = convert_pdf_to_images(pdf_path, os.path.join(output_dir, "images"))
    if len(image_paths) != 2:
        raise ValueError("PDF 必须包含至少两页")

    img1 = cv2.imread(image_paths[0])
    img2 = cv2.imread(image_paths[1])
    if img1 is None or img2 is None:
        raise ValueError("无法加载图像")

    # 找出相同区域，设置更高的容忍度
    mask = compare_and_mask_images(img1, img2, similarity_threshold=0.99)

    # 保存掩码
    mask_path = os.path.join(output_dir, "identical_mask.png")
    cv2.imwrite(mask_path, mask)
    # 保存模板图例
    module_img = img1.copy()
    module_img[mask != 255] = [255, 255, 255]
    module_img_path = os.path.join(output_dir, "module_img.png")
    cv2.imwrite(module_img_path, module_img)

    # 保存差异图像（用于调试）
    diff = cv2.absdiff(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    )
    cv2.imwrite(os.path.join(output_dir, "diff.png"), diff)

    # 遮挡相同区域
    masked_img1 = img1.copy()
    masked_img1[mask == 255] = [255, 255, 255]
    masked_img1_path = os.path.join(output_dir, "masked_page_1.png")
    cv2.imwrite(masked_img1_path, masked_img1)

    masked_img2 = img2.copy()
    masked_img2[mask == 255] = [255, 255, 255]
    masked_img2_path = os.path.join(output_dir, "masked_page_2.png")
    cv2.imwrite(masked_img2_path, masked_img2)

    print(f"相同区域掩码已保存至: {mask_path}")
    print(f"差异图像已保存至: {os.path.join(output_dir, 'diff.png')}")
    print(f"遮挡后的页面已保存至: {masked_img1_path}, {masked_img2_path}")


if __name__ == "__main__":
    pdf_path = "/home/star/zg/chat_doc_0630_online_cp/knowledge_base/保密知识库/layout/测试文档2/测试文档2/auto/测试文档2_origin.pdf"  # 替换为你的 PDF 文件路径
    output_dir = "/home/star/zg/MinerU/demo/output"  # 输出目录
    main(pdf_path, output_dir)
