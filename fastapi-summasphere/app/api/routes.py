from app.services.analyzer import TopicModelling
from app.services.summarizer import GeminiSummarizer
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter()

gemini_summarizer = GeminiSummarizer()
topic_modeller = TopicModelling()


@router.post("/summarize/gemini")
async def summarize_gemini(
    mode: str = Form(...),
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

        summary = gemini_summarizer.process_text(input_text, mode)
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@router.post("/analyzer")
async def analyze(
    media: str = Form(...),
    mode: str = Form(...),
    url: str = Form(None),
    file: UploadFile = File(None),
):
    try:
        input_text = None
        if url:
            input_text = url
        elif file:
            input_text = await file.read()
            # print(f"Read file content: {input_text[:100]}...")  # Debug PDF
        else:
            return JSONResponse(
                status_code=400, content={"message": "No input provided."}
            )

        analysis = topic_modeller.run_analysis(input_text, mode, media)
        return {"analysis": analysis}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
