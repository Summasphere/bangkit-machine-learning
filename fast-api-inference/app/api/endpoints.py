from fastapi import APIRouter, File, Form, UploadFile

from app.services.ai_processing import summarize_text
from app.services.file_handling import extract_text_from_pdf

router = APIRouter()


@router.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    extracted_text = await extract_text_from_pdf(file)
    summary = summarize_text(extracted_text)
    return {"filename": file.filename, "summary": summary}


@router.post("/summarize_text/")
async def summarize_text_endpoint(text: str = Form(...)):
    summary = summarize_text(text)
    return {"summary": summary}
