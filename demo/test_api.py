import requests

# API 端点
url = "http://localhost:5991/parse_doc"

# 准备上传的文件（假设有两个 PDF 文件）
files = [
    ("files", ("demo1.pdf", open("/home/cc099/MinerU/demo/pdfs/demo1.pdf", "rb"), "application/pdf")),
]

# 准备表单数据
data = {
    "output_dir": "/home/cc099/MinerU/demo/output_api",  # 输出目录
    "lang": "ch",                    # 文档语言
    "backend": "pipeline",           # 解析后端
    "method": "auto",                # 解析方法
    "start_page_id": 0,              # 开始页面
    "end_page_id": None,             # 结束页面（None 表示解析到文档末尾）
    "server_url": None,              # 如果使用 vlm-sglang-client，需要指定服务器 URL
}

try:
    # 发送 POST 请求
    response = requests.post(url, files=files, data=data)

    # 检查响应状态
    if response.status_code == 200:
        print("请求成功！")
        print(response.json())
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(response.json())
except Exception as e:
    print(f"请求过程中发生错误：{str(e)}")