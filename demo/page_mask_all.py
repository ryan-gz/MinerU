import cv2
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import os

def extract_template_regions(pdf_path, page_num, img, min_area=1000):
    """使用 pdfplumber 和图像处理提取潜在的模板区域（如页眉、页脚）"""
    regions = []
    with pdfplumber.open(pdf_path) as pdf:
        if page_num < len(pdf.pages):
            pdf_page = pdf.pages[page_num]
            # 提取页眉、页脚、表格等固定区域
            for rect in pdf_page.rects + pdf_page.images:
                x0, y0, x1, y1 = rect['x0'], rect['top'], rect['x1'], rect['bottom']
                area = (x1 - x0) * (y1 - y0)
                if area > min_area:  # 过滤小区域
                    regions.append((int(x0), int(y0), int(x1), int(y1)))

    # 使用边缘检测补充潜在模板区域
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            regions.append((x, y, x + w, y + h))

    return regions

def match_template(img, template, threshold=0.8):
    """在图像上匹配模板，返回掩码"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    loc = np.where(result >= threshold)
    h, w = gray_template.shape
    for pt in zip(*loc[::-1]):
        mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 255
    return mask

def process_pdf(pdf_path, output_pdf_path, template_page=0, match_threshold=0.8, min_area=1000):
    """处理 PDF，检测模板区域并生成掩码"""
    # 打开 PDF
    pdf_doc = fitz.open(pdf_path)
    images = []

    # 将每页转为图像
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        images.append(img)

    # 从指定页面提取模板区域
    template_regions = extract_template_regions(pdf_path, template_page, images[template_page], min_area)
    if not template_regions:
        print("No template regions found in the specified page.")
        pdf_doc.close()
        return

    # 对每页应用模板匹配
    output_pdf = fitz.open()
    for i, img in enumerate(images):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for (x0, y0, x1, y1) in template_regions:
            # 提取模板
            template = images[template_page][y0:y1, x0:x1]
            if template.size == 0:
                continue
            # 匹配模板
            temp_mask = match_template(img, template, match_threshold)
            mask = cv2.bitwise_or(mask, temp_mask)

        # 应用掩码
        masked_img = img.copy()
        masked_img[mask == 255] = [0, 0, 0]  # 模板区域设为黑色

        # 保存临时图像
        temp_path = f"temp_page_{i}.png"
        cv2.imwrite(temp_path, masked_img)
        # 添加到新 PDF
        page = output_pdf.new_page(width=img.shape[1], height=img.shape[0])
        page.insert_image(page.rect, filename=temp_path)
        os.remove(temp_path)

        # 保存掩码用于调试
        cv2.imwrite(f"mask_page_{i}.png", mask)

    output_pdf.save(output_pdf_path)
    output_pdf.close()
    pdf_doc.close()
    print(f"Output PDF saved to {output_pdf_path}")


if __name__ == "__main__":
    pdf_path = "/home/star/zg/chat_doc_0630_online_cp/knowledge_base/保密知识库/layout/测试文档2/测试文档2/auto/测试文档2_origin.pdf"  # 替换为你的 PDF 文件路径
    output_pdf_path = "/home/star/zg/MinerU/demo/output/masked.pdf"  # 输出目录
    process_pdf(pdf_path, output_pdf_path, template_page=0, match_threshold=0.8, min_area=1000)