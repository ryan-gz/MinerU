import os
import time
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pdf2image import convert_from_path
import img2pdf
from PIL import Image
import concurrent.futures
from functools import partial

def convert_pdf_to_images(pdf_path, output_dir, dpi=150):
    """将 PDF 转换为图像，使用较低的 DPI 以提高效率"""
    images = convert_from_path(pdf_path, dpi=dpi)  # 使用较低 DPI 减少图像大小
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

def compare_and_mask_images(
    img1, img2, window_size=80, stride=50, similarity_threshold=0.99
):
    # 检查图片是否成功加载
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return None

    # 检查图片尺寸是否一致，直接return None
    if img1.shape != img2.shape:
        return None

    # 获取图片尺寸
    height, width, _ = img1.shape

    # 创建掩码（初始化为全零）
    mask = np.zeros((height, width), dtype=np.uint8)

    # 预先转换为灰度图以减少每次滑窗的转换开销
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 滑窗逐行扫描，使用更大的步长以提高效率
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # 提取滑窗区域
            window1_gray = img1_gray[y : y + window_size, x : x + window_size]
            window2_gray = img2_gray[y : y + window_size, x : x + window_size]

            # 计算 SSIM 相似度
            similarity, _ = ssim(window1_gray, window2_gray, full=True)

            # 如果相似度超过阈值，标记为重叠区域
            if similarity >= similarity_threshold:
                mask[y : y + window_size, x : x + window_size] = 255
    return mask

def quick_similarity_check(img1_gray, img2_gray, sample_rate=0.1):
    """快速预检查两个页面是否足够相似，如果差异太大则跳过详细掩码计算"""
    # 采样部分像素进行快速比较
    height, width = img1_gray.shape
    samples = np.random.choice(height * width, size=int(height * width * sample_rate), replace=False)
    diff = np.abs(img1_gray.flatten()[samples] - img2_gray.flatten()[samples])
    mean_diff = np.mean(diff)
    return mean_diff < 50  # 如果平均像素差异小于阈值，才进行详细比较

def process_pair(i, image_paths):
    """处理一对相邻页面的函数，用于并行"""
    img1 = cv2.imread(image_paths[i])
    img2 = cv2.imread(image_paths[i+1])
    if img1 is None or img2 is None:
        return None, None, False

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 快速预检查
    if not quick_similarity_check(img1_gray, img2_gray):
        return img1, img2, False  # 无需掩码，直接返回原图

    mask = compare_and_mask_images(img1, img2)
    if mask is None or np.all(mask == 0):
        return img1, img2, False  # 无掩码

    # 应用掩码
    masked_img1 = img1.copy()
    masked_img1[mask == 255] = [255, 255, 255]
    masked_img2 = img2.copy()
    masked_img2[mask == 255] = [255, 255, 255]

    return masked_img1, masked_img2, True

def main(pdf_path, output_pdf_path, temp_dir="temp_images"):
    """主函数：处理整个 PDF，输出遮罩后的 PDF"""
    image_paths = convert_pdf_to_images(pdf_path, temp_dir)
    if len(image_paths) < 2:
        # 如果只有一页，直接转换为 PDF
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_paths))
        print(f"输出 PDF 已保存至: {output_pdf_path}")
        return

    processed_images = [None] * len(image_paths)
    masked_flags = [False] * (len(image_paths) - 1)

    # 使用线程池并行处理相邻页面对
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.cpu_count()/2)) as executor:
        process_func = partial(process_pair, image_paths=image_paths)
        results = list(executor.map(process_func, range(len(image_paths) - 1)))

    for i, (masked_img1, masked_img2, masked) in enumerate(results):
        if masked:
            processed_images[i] = masked_img1 if processed_images[i] is None else processed_images[i]
            processed_images[i+1] = masked_img2
        else:
            processed_images[i] = cv2.imread(image_paths[i]) if processed_images[i] is None else processed_images[i]
            processed_images[i+1] = cv2.imread(image_paths[i+1]) if processed_images[i+1] is None else processed_images[i+1]
        masked_flags[i] = masked

    # 处理最后一页（如果未处理）
    if processed_images[-1] is None:
        processed_images[-1] = cv2.imread(image_paths[-1])

    # 保存处理后的图像并转换为 PDF
    output_image_paths = []
    for i, img in enumerate(processed_images):
        if img is not None:
            path = os.path.join(temp_dir, f"processed_page_{i+1}.png")
            cv2.imwrite(path, img)
            output_image_paths.append(path)

    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(output_image_paths))

    # 清理临时文件
    for path in image_paths + output_image_paths:
        os.remove(path)
    os.rmdir(temp_dir)

    print(f"输出 PDF 已保存至: {output_pdf_path}")
    print(f"是否应用了掩码: {any(masked_flags)}")

if __name__ == "__main__":
    import glob
    for pdf_path in glob.glob("/home/star/zg/MinerU/demo/bm/*.pdf"):
        print(pdf_path)
        output_pdf_path = f"/home/star/zg/MinerU/demo/output_bm/{os.path.basename(pdf_path)[:-4]}_mask.pdf"  # 输出 PDF 路径
        main(pdf_path, output_pdf_path)


    # pdf_path = "/home/star/zg/chat_doc_0630_online_cp/knowledge_base/保密知识库/layout/测试文档5/测试文档5/auto/测试文档52_origin.pdf"  # 替换为你的 PDF 文件路径
    # output_pdf_path = "/home/star/zg/MinerU/demo/output/masked_output5.pdf"  # 输出 PDF 路径
    # main(pdf_path, output_pdf_path)