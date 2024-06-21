from app.services.analyzer import TopicModelling
from app.services.summarizer import GeminiSummarizer, BartSummarizer
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from ..utils.helpers import process_url

router = APIRouter()

gemini_summarizer = GeminiSummarizer()
topic_modeller = TopicModelling()
bart_summarizer = BartSummarizer() # bart_summarizer = None # change to none if you want to skip bart_summarizer

@router.post("/summarize")
async def summarize(
    mode: str = Form(...),          # (text, pdf, link)
    model: str = Form(None),        # (bart, gemini)
    text: str = Form(None),
    file: UploadFile = Form(None),
    url: str = Form(None),
):
    try:
        input_text = None
        if mode == "link":
            input_text = url
        elif mode == "pdf":
            input_text = await file.read()
        elif mode == "text":
            input_text = text

        if input_text is None:
            return JSONResponse(
                status_code=400, content={"message": "No input provided"}
            )

        if model and str(model).strip()=="bart":
            if mode == "pdf":
                input_text = gemini_summarizer.extract_text_from_pdf_buffer(input_text)
            elif mode == "link":
                input_text = process_url(input_text)
            summary = bart_summarizer.summarize(input_text)
        else:
            summary = gemini_summarizer.run_gemini_summarizer(input_text, mode)
        
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@router.post("/analyzer")
async def analyze(
    media: str = Form(...),     # (frontend, android)
    mode: str = Form(...),      # (pdf, link)
    url: str = Form(None),
    file: UploadFile = File(None),
):
    try:
        input_text = None
        if url and str(mode).strip()=="link":
            input_text = url
        elif file and str(mode).strip()=="pdf":
            input_text = await file.read()
        else:
            return JSONResponse(
                status_code=400, content={"message": "No input provided."}
            )

        analysis = topic_modeller.run_analysis(input_text, mode, media)
        return {"analysis": analysis}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
