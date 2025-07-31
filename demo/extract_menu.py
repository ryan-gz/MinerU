import os
import json
import glob


# 读取 json 文件
def read_json_file(file_path):
    # 按照 utf-8、gbk、gb2312、latin-1 的顺序读取文件
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return json.load(file)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"无法读取文件 {file_path}，请检查文件编码或格式。")


"""
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
"""


def extract_menu(json_data):
    menu = []
    current_level = 0
    level_stack = []

    for item in json_data:
        if "text_level" in item:
            text_level = item["text_level"]
            text = item["text"]
            page_idx = item.get("page_idx", "")

            # 确保 text_level 在 1 到 4 的范围内
            if 1 <= text_level <= 4:
                # 调整层级
                while len(level_stack) >= text_level:
                    level_stack.pop()
                level_stack.append((text, page_idx))

                # 构建目录条目
                entry = "    " * (text_level - 1) + f"{text} ... {page_idx}"
                menu.append(entry)

    return menu


def pprint_menu(menu):
    for entry in menu:
        print(entry)


json_path = "/home/cc099/MinerU/demo/output_long_pdf/中国辐射防护学会2024年学术年会论文集（中）_20240914143215/auto/中国辐射防护学会2024年学术年会论文集（中）_20240914143215_content_list.json"
menu = extract_menu(read_json_file(json_path))
pprint_menu(menu)

# # 看看整个测试集的menu 抽取效果
# menu_summary = []
# folder = "/home/cc099/MinerU/demo/output_test_database_qwen3-30b-origin-mineru"
# for json_path in glob.glob(f"{folder}/*/auto/*_content_list.json"):
#     f_name = os.path.basename(json_path).split("_content_list.json")[0]
#     menu = extract_menu(read_json_file(json_path))
#     menu_summary.append({"filename": f_name, "menu": "/n".join(menu)})
#     # 同时写入 txt
#     with open("/home/cc099/MinerU/demo/menu_summary.txt", "a", encoding="utf-8") as f:
#         f.write("*" * 50)
#         f.write(f"\nFILE_NAME: {f_name}\nMENU:\n")
#         for line in menu:
#             f.write(line + f"\n")
#         f.write(f"\n" * 2)


# # save menu_summary
# with open("/home/cc099/MinerU/demo/menu_summary.json", "w") as f:
#     json.dump(menu_summary, f, ensure_ascii=False, indent=4)
