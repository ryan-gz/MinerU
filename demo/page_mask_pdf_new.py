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
from tqdm import tqdm

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

def quick_similarity_check(img1_gray, img2_gray, sample_rate=0.1):
    """快速预检查两个页面是否足够相似，如果差异太大则跳过详细掩码计算"""
    height, width = img1_gray.shape
    samples = np.random.choice(height * width, size=int(height * width * sample_rate), replace=False)
    diff = np.abs(img1_gray.flatten()[samples] - img2_gray.flatten()[samples])
    mean_diff = np.mean(diff)
    return mean_diff < 50  # 如果平均像素差异小于阈值，才进行详细比较

def compare_and_mask_images(
    img1, img2, window_size=80, stride=50, similarity_threshold=0.99
):
    # 检查图片是否成功加载
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return None

    # 检查图片尺寸是否一致，如果不一致，调整大小
    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))

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

def find_similar_pairs(image_paths, gray_images):
    """步骤1: 快速找出所有相邻相似的页面对"""
    similar_pairs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        def check_pair(i):
            img1_gray = gray_images[i]
            img2_gray = gray_images[i+1]
            if quick_similarity_check(img1_gray, img2_gray):
                return (i, i+1)
            return None
        
        # 使用 tqdm 显示相似对查找进度
        results = list(tqdm(executor.map(check_pair, range(len(image_paths) - 1)), total=len(image_paths) - 1, desc="Finding similar pairs"))
    
    similar_pairs = [pair for pair in results if pair is not None]
    return similar_pairs

def check_page_against_pattern(page_gray, pattern_template, sample_rate=0.1, strict_threshold=50):
    """使用quick check思路检查页面是否匹配pattern: 计算mean_diff，严格阈值"""
    height, width = page_gray.shape
    samples = np.random.choice(height * width, size=int(height * width * sample_rate), replace=False)
    diff = np.abs(page_gray.flatten()[samples] - pattern_template.flatten()[samples])
    mean_diff = np.mean(diff)
    return mean_diff < strict_threshold

def apply_mask_to_page(img, mask):
    """应用mask到页面: mask区域设置为白色"""
    masked_img = img.copy()
    masked_img[mask == 255] = [255, 255, 255]
    return masked_img

def apply_mask_fold_to_page(img, mask):
    """应用mask到页面: 非mask区域设置为白色"""
    masked_fold = img.copy()
    masked_fold[mask != 255] = [255, 255, 255]
    return masked_fold

def main(pdf_path, output_pdf_path, temp_dir="temp_images"):
    """优化后的主函数"""
    image_paths = convert_pdf_to_images(pdf_path, temp_dir)
    if len(image_paths) < 2:
        # 如果只有一页，直接转换为 PDF
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_paths))
        print(f"输出 PDF 已保存至: {output_pdf_path}")
        return

    # 预加载所有图像和灰度图
    images = [cv2.imread(path) for path in image_paths]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    print("Starting to find similar pairs...")
    # 步骤1: 快速找出相似页面对
    similar_pairs = find_similar_pairs(image_paths, gray_images)
    print(f"Found {len(similar_pairs)} similar pairs.")

    if not similar_pairs:
        # 没有相似对，直接输出原PDF图像
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_paths))
        print(f"输出 PDF 已保存至: {output_pdf_path} (无掩码应用)")
        return

    print("Starting to search for mask pattern in similar pairs...")
    # 步骤2: 遍历相似对，尝试寻找第一个有效的mask pattern
    mask_pattern = None
    pattern_template = None
    used_pair = None

    for pair in similar_pairs:
        i, j = pair
        img1, img2 = images[i], images[j]
        mask = compare_and_mask_images(img1, img2)
        if mask is not None and np.any(mask == 255):  # 找到有效的mask
            mask_pattern = mask
            # 使用img1的灰度创建pattern模板 (非mask区域黑色)
            pattern_template = apply_mask_fold_to_page(img1, mask)
            used_pair = pair
            print(f"Found valid mask pattern from pair {used_pair}")
            break  # 找到第一个就停止
        else:
            print(f"No valid mask found in pair {pair}")

    if mask_pattern is None:
        print("No valid mask pattern found in any pair.")
        # 没有找到mask pattern，直接输出
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_paths))
        print(f"输出 PDF 已保存至: {output_pdf_path} (无掩码应用)")
        return

    print("Starting to check all pages against the mask pattern...")
    # 步骤3: 对所有页面确认是否匹配这个pattern
    processed_images = images.copy()
    masked_flags = [False] * len(images)
    pattern_template_gray = cv2.cvtColor(pattern_template, cv2.COLOR_BGR2GRAY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        def check_and_mask(k):
            image_mask_fold = apply_mask_fold_to_page(images[k], mask_pattern)
            page_gray = cv2.cvtColor(image_mask_fold, cv2.COLOR_BGR2GRAY)
            if check_page_against_pattern(page_gray, pattern_template_gray):
                masked_img = apply_mask_to_page(images[k], mask_pattern)
                return masked_img, True
            return images[k], False
        
        # 使用 tqdm 显示逐页检查进度
        results = list(tqdm(executor.map(check_and_mask, range(len(images))), total=len(images), desc="Checking pages"))

    processed_images = [res[0] for res in results]
    masked_flags = [res[1] for res in results]

    # 保存处理后的图像并转换为 PDF
    output_image_paths = []
    for i, img in enumerate(processed_images):
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
    if mask_pattern is not None:
        print(f"使用了页面对 {used_pair} 的mask pattern")

if __name__ == "__main__":
    pdf_path = "/home/star/zg/chat_doc_0630_online_cp/knowledge_base/保密知识库/layout/测试文档5/测试文档5/auto/测试文档5_origin.pdf"  # 替换为你的 PDF 文件路径
    output_pdf_path = "/home/star/zg/MinerU/demo/output/masked_output5.pdf"  # 输出 PDF 路径
    main(pdf_path, output_pdf_path)