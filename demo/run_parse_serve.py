import sys

sys.path.append(".")
import copy
import json
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
from loguru import logger
import tqdm
from mineru_omni.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    prepare_env,
    read_fn,
)
from mineru_omni.data.data_reader_writer import FileBasedDataWriter
from mineru_omni.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru_omni.utils.enum_class import MakeMode
from mineru_omni.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru_omni.backend.pipeline.pipeline_analyze import (
    doc_analyze as pipeline_doc_analyze,
)
from mineru_omni.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru_omni.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from mineru_omni.backend.vlm.vlm_middle_json_mkcontent import (
    union_make as vlm_union_make,
)

os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
# FastAPI app instance
app = FastAPI(title="Document Parsing API", description="API for parsing PDF documents using Mineru Omni")

# Include the original do_parse function here (unchanged)
def do_parse(
    output_dir,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    p_lang_list: list[str],
    backend="pipeline",
    parse_method="auto",
    formula_enable=True,
    table_enable=True,
    server_url=None,
    f_draw_layout_bbox=True,
    f_draw_span_bbox=True,
    f_dump_md=True,
    f_dump_middle_json=True,
    f_dump_model_output=True,
    f_dump_orig_pdf=True,
    f_dump_content_list=True,
    f_make_md_mode=MakeMode.MM_MD,
    start_page_id=0,
    end_page_id=None,
):
    # ... (The original do_parse function code, copied verbatim)
    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                pdf_bytes_list,
                p_lang_list,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        )

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = (
                FileBasedDataWriter(local_image_dir),
                FileBasedDataWriter(local_md_dir),
            )

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(
                model_list,
                images_list,
                pdf_doc,
                image_writer,
                _lang,
                _ocr_enable,
                formula_enable,
            )

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                )

            if f_draw_span_bbox:
                draw_span_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                )

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(
                    pdf_info, f_make_md_mode, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(
                    pdf_info, MakeMode.CONTENT_LIST, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = (
                FileBasedDataWriter(local_image_dir),
                FileBasedDataWriter(local_md_dir),
            )
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=backend,
                server_url=server_url,
            )

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                )

            if f_draw_span_bbox:
                draw_span_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                )

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(
                    pdf_info, MakeMode.CONTENT_LIST, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")

@app.post("/parse_doc", summary="Parse documents and generate output files")
async def parse_doc_api(
    files: List[UploadFile] = File(...),
    output_dir: str = Form(...),
    lang: str = Form(default="ch"),
    backend: str = Form(default="pipeline"),
    method: str = Form(default="auto"),
    server_url: Optional[str] = Form(default=None),
    start_page_id: int = Form(default=0),
    end_page_id: Optional[int] = Form(default=None),
):
    """
    API endpoint to parse documents (PDFs or images) and generate output files.

    Parameters:
    - files: List of uploaded files (PDFs or images).
    - output_dir: Directory to store parsing results.
    - lang: Language of the documents (default: 'ch'). Options: ['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka'].
    - backend: Parsing backend (default: 'pipeline'). Options: ['pipeline', 'vlm-transformers', 'vlm-sglang-engine', 'vlm-sglang-client'].
    - method: Parsing method (default: 'auto'). Options: ['auto', 'txt', 'ocr']. Only applicable for 'pipeline' backend.
    - server_url: Server URL for 'vlm-sglang-client' backend (e.g., 'http://127.0.0.1:30000').
    - start_page_id: Start page ID for parsing (default: 0).
    - end_page_id: End page ID for parsing (default: None, parse until end).

    Returns:
    - JSON response with status and output directory information.
    """
    try:
        # Create a temporary directory to store uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            path_list = []

            # Save uploaded files to temporary directory
            for file in files:
                file_name = file.filename
                temp_file_path = os.path.join(temp_dir, file_name)
                with open(temp_file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                file_name_list.append(Path(file_name).stem)
                pdf_bytes_list.append(content)
                lang_list.append(lang)
                path_list.append(Path(temp_file_path))

            # Call the original parse_doc function
            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": f"Documents parsed successfully. Results saved to {output_dir}",
                    "output_dir": output_dir,
                },
            )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error parsing documents: {str(e)}")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5991)