# Copyright (c) Opendatalab. All rights reserved.
import re
import json
import json_repair
from loguru import logger
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any

from mineru_omni.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text


class DictOutputParser(BaseOutputParser):
    """
    A LangChain-compatible output parser that extracts all {...} dictionaries from a string.
    """    
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the input text and return a list of dictionaries.
        
        Args:
            text (str): Input string containing {...} patterns
            
        Returns:
            List[Dict[str, Any]]: List of parsed dictionaries
        """
        pattern = r'\{.*?\}'
        matches = re.search(pattern, text, re.DOTALL)
        result = json_repair.loads(matches.group(0))      
        return result
    
    @property
    def _type(self) -> str:
        """Return the parser type for LangChain compatibility."""
        return "dict_output_parser"
    

PROMPT_TEMPLATE = """输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑
    - 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们

IMPORTANT: 
请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不需要对字典格式化，不需要返回任何其他信息。

Input title list:
{title_dict}

Corrected title list:
"""

def get_title_dict(page_info_list):
    title_dict = {}
    origin_title_list = []
    # 遍历页面信息列表
    i = 0
    for page_info in page_info_list:
        blocks = page_info["para_blocks"]
        for block in blocks:
            # 如果块类型为标题
            if block["type"] == "title":
                # 将块添加到原始标题列表
                origin_title_list.append(block)
                # 合并块中的文本
                title_text = merge_para_with_text(block)

                if 'line_avg_height' in block:
                    line_avg_height = block['line_avg_height']
                else:
                # 否则，计算块中每行的行高，并取平均值
                    title_block_line_height_list = []
                    for line in block['lines']:
                        bbox = line['bbox']
                        title_block_line_height_list.append(int(bbox[3] - bbox[1]))
                    if len(title_block_line_height_list) > 0:
                        line_avg_height = sum(title_block_line_height_list) / len(title_block_line_height_list)
                    else:
                    # 如果块中没有行高信息，则取块的高度
                        line_avg_height = int(block['bbox'][3] - block['bbox'][1])

                title_dict[f"{i}"] = [title_text, line_avg_height, int(page_info['page_idx']) + 1]
                i += 1
    return title_dict, origin_title_list

def split_dict(d, max_len):
    """
    将字典分割成多个小字典，确保每个小字典的 str() 长度不超过 max_len
    
    Args:
        d (dict): 输入字典
        max_len (int): 每个小字典字符串表示的最大长度
    
    Returns:
        list: 包含多个小字典的列表
    """
    result = []
    current_dict = {}
    current_len = 0
    
    # 用于计算字典的字符串长度
    def get_dict_str_len(d):
        return len(str(d))
    
    for key, value in d.items():
        # 尝试添加新键值对
        temp_dict = current_dict.copy()
        temp_dict[key] = value
        
        # 检查添加后的长度
        temp_len = get_dict_str_len(temp_dict)
        
        if temp_len <= max_len:
            # 如果长度符合要求，直接添加到当前字典
            current_dict = temp_dict
            current_len = temp_len
        else:
            # 如果当前字典不为空，先保存
            if current_dict:
                result.append(current_dict)
            # 创建新字典只包含当前键值对
            current_dict = {key: value}
            current_len = get_dict_str_len(current_dict)
    
    # 添加最后一个字典（如果不为空）
    if current_dict:
        result.append(current_dict)
    
    return result

def llm_aided_title_omni(page_info_list, title_aided_config):
    def infer_title(tmp_title_dict, chain, max_retries=3):
        retry_count = 0
        while retry_count < max_retries:
            try:
                tmp_dict_completion = chain.invoke({
                        "title_dict": tmp_title_dict
                    })
                # 如果生成的标题优化结果中包含"</think>"，则去掉"</think>"及其之前的内容
                tmp_dict_completion = {int(k): int(v) for k, v in tmp_dict_completion.items()}
                # 将生成的标题优化结果转换为字典
                if len(tmp_dict_completion) == len(tmp_title_dict):
                    return tmp_dict_completion
                else:
                    logger.warning(
                        "The number of titles in the optimized result is not equal to the number of titles in the input.")
                    retry_count += 1
            except Exception as e:
                # 否则，记录警告信息，并增加重试次数
                logger.exception(e)
                retry_count += 1
    
    llm = ChatOpenAI(
        model=title_aided_config["model"],
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
        temperature=0.7,
        # max_tokens=512,
        # model_kwargs={"repetition_penalty": 1.05}
    )
    prompt = ChatPromptTemplate.from_messages([("user", PROMPT_TEMPLATE)])
    parser = DictOutputParser()
    chain = prompt | llm | parser
    title_dict,origin_title_list = get_title_dict(page_info_list)
    max_retries = 5
    dict_completion = None
    max_len = 1024*5  # 设置最大长度限制
    s_title_dict_list = split_dict(title_dict, max_len)  # 将字典分割成多个小字典
    for s_title_dict in s_title_dict_list:
        s_dict_completion = infer_title(s_title_dict, chain, max_retries)
        # 合并 s_dict_completion 到 dict_completion
        if dict_completion is None:
            dict_completion = s_dict_completion
        else:
            dict_completion.update(s_dict_completion)

    if dict_completion is None:
        # 如果发生异常，记录异常信息，并增加重试次数
        logger.error("Failed to decode dict after maximum retries.")
    else:
        if len(dict_completion) == len(title_dict):
            for i, origin_title_block in enumerate(origin_title_list):
                # 没有 return 而是直接通过修改
                origin_title_block["level"] = int(dict_completion[i])


def llm_aided_title(page_info_list, title_aided_config):
    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )
    title_dict = {}
    origin_title_list = []
    # 遍历页面信息列表
    i = 0
    for page_info in page_info_list:
        blocks = page_info["para_blocks"]
        for block in blocks:
            # 如果块类型为标题
            if block["type"] == "title":
                # 将块添加到原始标题列表
                origin_title_list.append(block)
                # 合并块中的文本
                title_text = merge_para_with_text(block)

                if 'line_avg_height' in block:
                    line_avg_height = block['line_avg_height']
                else:
                # 否则，计算块中每行的行高，并取平均值
                    title_block_line_height_list = []
                    for line in block['lines']:
                        bbox = line['bbox']
                        title_block_line_height_list.append(int(bbox[3] - bbox[1]))
                    if len(title_block_line_height_list) > 0:
                        line_avg_height = sum(title_block_line_height_list) / len(title_block_line_height_list)
                    else:
                    # 如果块中没有行高信息，则取块的高度
                        line_avg_height = int(block['bbox'][3] - block['bbox'][1])

                title_dict[f"{i}"] = [title_text, line_avg_height, int(page_info['page_idx']) + 1]
                i += 1
    # logger.info(f"Title list: {title_dict}")

    title_optimize_prompt = f"""输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑
    - 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们

IMPORTANT: 
请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不需要对字典格式化，不需要返回任何其他信息。

Input title list:
{title_dict}

Corrected title list:
"""

    retry_count = 0
    max_retries = 3
    dict_completion = None
    # 2025-07-18 增加对 长文本的适应性
    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=title_aided_config["model"],
                messages=[
                    {'role': 'user', 'content': title_optimize_prompt}],
                temperature=0.7,
                stream=True,
            )
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)
            # 将生成的标题优化结果拼接成字符串
            content = "".join(content_pieces).strip()
            # logger.info(f"Title completion: {content}")
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()
            dict_completion = json_repair.loads(content) # 使用 json_repair 修复字符串
            # 如果生成的标题优化结果中包含"</think>"，则去掉"</think>"及其之前的内容
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            # logger.info(f"len(dict_completion): {len(dict_completion)}, len(title_dict): {len(title_dict)}")
            # 将生成的标题优化结果转换为字典
            if len(dict_completion) == len(title_dict):
                for i, origin_title_block in enumerate(origin_title_list):
                    # 没有 return 而是直接通过修改
                    origin_title_block["level"] = int(dict_completion[i])
                break
            # 如果生成的标题优化结果中的标题数量和输入的标题数量一致，则将生成的标题优化结果应用到原始标题列表中
            else:
                logger.warning(
                    "The number of titles in the optimized result is not equal to the number of titles in the input.")
                retry_count += 1
        except Exception as e:
            # 否则，记录警告信息，并增加重试次数
            logger.exception(e)
            retry_count += 1

    if dict_completion is None:
        # 如果发生异常，记录异常信息，并增加重试次数
        logger.error("Failed to decode dict after maximum retries.")
    # 如果达到最大重试次数后仍未成功，则记录错误信息
