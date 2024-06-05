from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.ai_processing import summarize_text
from app.services.file_handling import extract_text_from_pdf

router = APIRouter()


@router.post("/summarize/")
async def summarize(
    text: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)
):
    if text:
        result = summarize_text(text)
        return {"title": result["title"], "summary": result["summary"]}
    elif file:
        extracted_text = await extract_text_from_pdf(file)
        result = summarize_text(extracted_text)
        return {"title": result["title"], "summary": result["summary"]}
    else:
        raise HTTPException(status_code=400, detail="No text or file provided")
