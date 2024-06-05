from io import BytesIO

import PyPDF2
from fastapi import UploadFile


async def extract_text_from_pdf(upload_file: UploadFile) -> str:
    content = await upload_file.read()
    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
