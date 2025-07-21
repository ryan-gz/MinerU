import json

# 读取 json 文件
def read_json_file(file_path):
    # 按照 utf-8、gbk、gb2312、latin-1 的顺序读取文件
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return json.load(file)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue    
    raise ValueError(f"无法读取文件 {file_path}，请检查文件编码或格式。")

'''
json 内容实例，提取出 val中带 text_level 的内容,text_level从1到4，代表1级标题到4级标题，整理成目录的样式
[
    {
        "type": "text",
        "text": "《亚运服务保障手册》编审小组",
        "text_level": 3,
        "page_idx": 12
    },
]
目录样式
一级标题 ... 页码
    二级标题 ... 页码
        三级标题 ... 页码
            四级标题 ... 页码
    
    二级标题 ... 页码
        三级标题 ... 页码
一级标题 ... 页码
'''
def extract_menu(json_data):
    menu = []
    current_level = 0
    level_stack = []

    for item in json_data:
        if 'text_level' in item:
            text_level = item['text_level']
            text = item['text']
            page_idx = item.get('page_idx', '')

            # 确保 text_level 在 1 到 4 的范围内
            if 1 <= text_level <= 4:
                # 调整层级
                while len(level_stack) >= text_level:
                    level_stack.pop()
                level_stack.append((text, page_idx))

                # 构建目录条目
                entry = '    ' * (text_level - 1) + f"{text} ... {page_idx}"
                menu.append(entry)

    return menu

def pprint_menu(menu):
    for entry in menu:
        print(entry)

json_path = "/home/cc099/MinerU/demo/output_test_database/b63d16c05c614694a7adde123be7cb4d/auto/b63d16c05c614694a7adde123be7cb4d_content_list.json"
menu = extract_menu(read_json_file(json_path))
pprint_menu(menu)   