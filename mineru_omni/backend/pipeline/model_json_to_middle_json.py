# Copyright (c) Opendatalab. All rights reserved.
import os
import time
import base64
import requests

from loguru import logger
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from mineru_omni.utils.config_reader import get_device, get_llm_aided_config, get_formula_enable
from mineru_omni.backend.pipeline.model_init import AtomModelSingleton
from mineru_omni.backend.pipeline.para_split import para_split
from mineru_omni.utils.block_pre_proc import prepare_block_bboxes, process_groups
from mineru_omni.utils.block_sort import sort_blocks_by_bbox
from mineru_omni.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru_omni.utils.cut_image import cut_image_and_table
from mineru_omni.utils.enum_class import ContentType
from mineru_omni.utils.llm_aided import llm_aided_title,llm_aided_title_omni
from mineru_omni.utils.model_utils import clean_memory
from mineru_omni.backend.pipeline.pipeline_magic_model import MagicModel
from mineru_omni.utils.ocr_utils import OcrConfidence
from mineru_omni.utils.span_block_fix import fill_spans_in_blocks, fix_discarded_block, fix_block_spans
from mineru_omni.utils.span_pre_proc import remove_outside_spans, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, txt_spans_extract
from mineru_omni.version import __version__
from mineru_omni.utils.hash_utils import str_md5


# resnet
IMG_SIZE = 224
CLASS_MODEL = "/mnt/ddata2/user/zhangga/omniknow/stash/flowchart/weights/classify/flowchart_classifier_epoch_1_acc_93.37.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vlm
VLLM_ENDPOINT = "http://192.168.5.210:8533/v1/chat/completions"
VLM_MODEL_NAME = "mermaid"


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False, formula_enabled=True):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = str_md5(image_dict["img_base64"])
    page_w, page_h = map(int, page.get_size())
    magic_model = MagicModel(page_model_info, scale)

    """从magic_model对象中获取后面会用到的区块信息"""
    discarded_blocks = magic_model.get_discarded()
    text_blocks = magic_model.get_text_blocks()
    title_blocks = magic_model.get_title_blocks()
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations()

    img_groups = magic_model.get_imgs()
    table_groups = magic_model.get_tables()

    """对image和table的区块分组"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks, maybe_text_image_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks, _ = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    """获取所有的spans信息"""
    spans = magic_model.get_all_spans()

    """某些图可能是文本块，通过简单的规则判断一下"""
    if len(maybe_text_image_blocks) > 0:
        for block in maybe_text_image_blocks:
            span_in_block_list = []
            for span in spans:
                if span['type'] == 'text' and calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block['bbox']) > 0.7:
                    span_in_block_list.append(span)
            if len(span_in_block_list) > 0:
                # span_in_block_list中所有bbox的面积之和
                spans_area = sum((span['bbox'][2] - span['bbox'][0]) * (span['bbox'][3] - span['bbox'][1]) for span in span_in_block_list)
                # 求ocr_res_area和res的面积的比值
                block_area = (block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1])
                if block_area > 0:
                    ratio = spans_area / block_area
                    if ratio > 0.25 and ocr_enable:
                        # 移除block的group_id
                        block.pop('group_id', None)
                        # 符合文本图的条件就把块加入到文本块列表中
                        text_blocks.append(block)
                    else:
                        # 如果不符合文本图的条件，就把块加回到图片块列表中
                        img_body_blocks.append(block)
            else:
                img_body_blocks.append(block)


    """将所有区块的bbox整理到一起"""
    if formula_enabled:
        interline_equation_blocks = []

    if len(interline_equation_blocks) > 0:

        for block in interline_equation_blocks:
            spans.append({
                "type": ContentType.INTERLINE_EQUATION,
                'score': block['score'],
                "bbox": block['bbox'],
            })

        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    """在删除重复span之前，应该通过image_body和table_body的block过滤一下image和table的span"""
    """顺便删除大水印并保留abandon的span"""
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    """删除重叠spans中置信度较低的那些"""
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    """删除重叠spans中较小的那些"""
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """根据parse_mode，构造spans，主要是文本类的字符填充"""
    if ocr_enable:
        pass
    else:
        """使用新版本的混合ocr方案."""
        spans = txt_spans_extract(page, spans, page_pil_img, scale, all_bboxes, all_discarded_blocks)

    """先处理不需要排版的discarded_blocks"""
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """如果当前页面没有有效的bbox则跳过"""
    if len(all_bboxes) == 0:
        return None

    """对image/table/interline_equation截图"""
    for span in spans:
        if span['type'] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    """span填充进block"""
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)

    """对block进行fix操作"""
    fix_blocks = fix_block_spans(block_with_spans)

    """对block进行排序"""
    sorted_blocks = sort_blocks_by_bbox(fix_blocks, page_w, page_h, footnote_blocks)

    """构造page_info"""
    page_info = make_page_info_dict(sorted_blocks, page_index, page_w, page_h, fix_discarded_blocks)

    return page_info


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enabled=True):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    formula_enabled = get_formula_enable(formula_enabled)
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc="Processing pages"):
        page = pdf_doc[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page, image_writer, page_index, ocr_enable=ocr_enable, formula_enabled=formula_enabled
        )
        if page_info is None:
            page_w, page_h = map(int, page.get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)

    """后置ocr处理"""
    need_ocr_list = []
    img_crop_list = []
    text_block_list = []
    for page_info in middle_json["pdf_info"]:
        for block in page_info['preproc_blocks']:
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)
    for block in text_block_list:
        for line in block['lines']:
            for span in line['spans']:
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(span['np_img'])
                    span.pop('np_img')
    if len(img_crop_list) > 0:
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang=lang
        )
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0

    """分段"""
    para_split(middle_json["pdf_info"])

    """llm优化"""
    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        """标题优化"""
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None:
            if title_aided_config.get('enable', False):
                llm_aided_title_start_time = time.time()
                # llm_aided_title(middle_json["pdf_info"], title_aided_config)
                llm_aided_title_omni(middle_json["pdf_info"], title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    """清理内存"""
    pdf_doc.close()
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json


def result_to_middle_json_omni(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enabled=True):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    formula_enabled = get_formula_enable(formula_enabled)
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc="Processing pages"):
        page = pdf_doc[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page, image_writer, page_index, ocr_enable=ocr_enable, formula_enabled=formula_enabled
        )
        if page_info is None:
            page_w, page_h = map(int, page.get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)

    """后置ocr处理"""
    need_ocr_list = []
    img_crop_list = []
    text_block_list = []
    for page_info in middle_json["pdf_info"]:
        for block in page_info['preproc_blocks']:
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
                    # 为 image_body 增加 content, 文生图为流程图 mermaid 代码
                    if sub_block['type'] == 'image_body':
                        for line in sub_block['lines']:
                            for span in line['spans']:
                                if span.get('image_path'):
                                    # tomo: 使用微调的 resnet18n 做二分类
                                    img_path = os.path.join(image_writer._parent_dir,span['image_path'])
                                    is_flowchart = predict_classify(filepath=img_path, model_path=CLASS_MODEL)
                                    
                                    if is_flowchart:
                                        # tomo: 使用微调的 qwenvl2.5-3b 实现 img2mermaid
                                        resp = process_image(image_path=img_path, vlm_endpoint=VLLM_ENDPOINT, model=VLM_MODEL_NAME)
                                        if resp['status'] == 'success':
                                            span['content'] = resp['mermaid_code']





            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)
    for block in text_block_list:
        for line in block['lines']:
            for span in line['spans']:
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(span['np_img'])
                    span.pop('np_img')
    if len(img_crop_list) > 0:
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.3,
            lang=lang
        )
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0

    """分段"""
    para_split(middle_json["pdf_info"])

    """llm优化"""
    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        """标题优化"""
        title_aided_config = llm_aided_config.get('title_aided', None)
        if title_aided_config is not None:
            if title_aided_config.get('enable', False):
                llm_aided_title_start_time = time.time()
                # llm_aided_title(middle_json["pdf_info"], title_aided_config)
                llm_aided_title_omni(middle_json["pdf_info"], title_aided_config)
                logger.info(f'llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}')

    """清理内存"""
    pdf_doc.close()
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json



def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict





def predict_classify(filepath=None, model_path=None):
    """
    预测入口函数

    参数:
        filepath: 单张图像的路径 (可选)
        folderpath: 图像文件夹的路径 (可选)
        model_path: 模型文件的路径 (必须)
    """

    def load_model(model_path):
        """加载训练好的模型"""
        # 创建模型结构
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # 二分类

        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()  # 设置为评估模式
        return model

    def preprocess_image(image_path):
        """预处理输入图像"""
        transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # 添加批次维度
        return image.to(DEVICE)

    def predict_image(model, image_path):
        """预测单张图像是否为流程图"""
        try:
            # 预处理图像
            image = preprocess_image(image_path)

            # 预测
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                confidence = torch.softmax(outputs.data, 1)[0][predicted.item()].item() * 100

            result = predicted.item()

            return {
                "filename": os.path.basename(image_path),
                "result": result,
                "confidence": confidence,
                "success": True,
            }
        except Exception as e:
            return {"filename": os.path.basename(image_path), "error": str(e), "success": False}

    # 加载模型
    model = load_model(model_path)

    # 预测单张图像
    if not os.path.exists(filepath):
        print(f"错误: 图像文件不存在 - {filepath}")
        return False

    result = predict_image(model, filepath)
    if result["success"]:
        return result["result"]
    else:
        print(f"处理 {result['filename']} 时出错: {result['error']}")
        return False

def replace_chinese_symbols(text):
    # 创建中文符号到英文符号的映射字典
    symbol_map = {
        "，": ",",
        "。": ".",
        "、": ",",
        "；": ";",
        "：": ":",
        "？": "?",
        "！": "!",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "—": "-",
        "…": "...",
        "～": "~",
    }

    # 遍历映射字典，替换文本中的符号
    for chinese, english in symbol_map.items():
        text = text.replace(chinese, english)
    text = text.replace('"', "").replace("'", "")
    return text


def process_image(image_path, vlm_endpoint, model):
    """处理单张图片，返回生成的Mermaid代码"""
    prompt = """
    你是 Mermaid 语法生成专家，需要将这张图片的结构化信息转为标准 Mermaid 代码。
    规则：
    1. 图表类型严格遵循 mermaid 语法（如需调整节点位置，可添加方向标识如 -->|条件|）；
    2. 仅输出 Mermaid 代码（无需额外解释），代码需可直接在 Mermaid 编辑器中运行；
    3. 节点命名简洁，关系清晰，避免冗余信息；
    4. 若 VLM 信息包含条件分支（如“成功/失败”），用 -->|条件| 表示。
    """

    try:
        # 将图片转为Base64编码
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"

        # 构造多模态请求体
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 10240,
            "temperature": 0.2,
        }

        # 发送请求并解析响应
        headers = {"Content-Type": "application/json"}
        response = requests.post(vlm_endpoint, headers=headers, json=payload)
        response.raise_for_status()  # 检查HTTP错误

        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        answer = replace_chinese_symbols(answer)

        return {"status": "success", "image_path": image_path, "mermaid_code": answer}

    except Exception as e:
        return {"status": "error", "image_path": image_path, "error": str(e)}