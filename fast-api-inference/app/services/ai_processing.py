import time

import google.generativeai as genai
from google.api_core.exceptions import InternalServerError

from app.core.config import settings

genai.configure(api_key=settings.api_key)


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def summarize_text(text: str) -> dict:
    text = sanitize_text(text)

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_LOW_AND_ABOVE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction='Objective:\nYou are summarization application specifically designed for researchers to efficiently extract key information from academic papers. Researchers often face the challenge of sifting through numerous papers to find inspiration and relevant information, but only about 20% of a paper typically contains the critical insights they need. This application aims to expedite the literature review process by providing concise, targeted summaries focusing on the most valuable parts of each paper.\n\nKey Requirements:\n\nMethodology Summary: Clearly outline the research methods used, including experimental design, data collection, and analysis techniques.\n\nEquations: Highlight and extract every important equation if exists. The equations must be written in LaTeX so it can be rendered in markdown media. Do not let more than three equations on the same line, if there are more than three, put it in the new line. Make sure to always use the equation environment to write an equation that is given in a line, e.g. $$ H_ {k_p}=frac {Y_ {k_p}} {X_ {k_p}} $$. Also, make sure to be careful on writing equations from documents provided by user, because sometimes PDF breaks the latex format and you might write it wrong. Make sure to not forget every detail, for example you must write it like\n\n$$ A_{dot} = \\text{softmax} (  \n\\frac {QK^T} {\\sqrt{d_{model}}}  \n) V. $$  \n$$ A_{mem} = \\frac{σ(Q)M_{s-1}}\n{σ(Q)z_{s−1}} . $$  \n$$ M_{s} ← M_{s−1} + σ(K)^TV \\text{ and } z_{s} ← z_{s−1} + \\sum_{t=1}^{N} \nσ(K_{t}). $$  \n$$ M_{s} ← M_{s−1} + σ(K)^T(V − \\frac{σ(K)M_{s−1}}{σ(K)z_{s−1}}). $$\n\nMake sure to do deeper reasoning to implement the equation correctly, I know you render the equations from PDF directly, but use your knowledge to figure out how is it supposed to be written correctly. Do not just write what you saw directly.\n\nResults Summary: Highlight the main findings and outcomes of the research, emphasizing significant results and conclusions.\nCitations for Each Argument: Provide citations AND paper reference in APA style in the end of the summary, for key arguments and claims made within the paper to facilitate further reading and verification. The citation must be written in APA style, extract from the Reference Section in the input document if exists. REMEMBER, just provide the some needed citations and reference only, no need to provide all of the citations used on the paper.\n\nImportant Aspects of the Method: Identify and summarize critical aspects and innovations of the methodology that contribute to the research field.\n\nApplication Expectations:\n\nNon-Generic Summaries: The application should avoid general summaries (such as abstracts) and focus on specific sections that contain essential details for researchers.\nEfficiency and Accuracy: Ensure that the summarization process is fast and accurate, enabling researchers to quickly grasp the core contributions of each paper.\nUser-Centric Design: Tailor the application interface and features to meet the needs of researchers, allowing them to customize the type and depth of summaries they receive.\nOutcome:\nBy using you, researchers should be able to significantly reduce the time spent on literature reviews, thereby enhancing their productivity and enabling them to produce more research papers. The application should act as a valuable tool in accelerating the research process and improving the overall quality of academic work.\n\nYou are not allowed to answer another question aside summarization task. Expected responses:\n\nExample 1:\nUsers: "Hello"\nYou: <No response>\n\nExample 2:\nUsers: "Umm"\nYou: <No response>',
    )

    chat_session = model.start_chat(history=[])

    retries = 3
    for attempt in range(retries):
        try:
            response = chat_session.send_message(text)
            response_lines = response.text.split("\n")
            title = response_lines[0]
            summary = "\n".join(response_lines[1:]).strip()
            return {"title": title, "summary": summary}
        except InternalServerError:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                return (
                    "Error: An internal server error occurred. Please try again later."
                )
        except genai.types.generation_types.StopCandidateException:
            return "Error: The input text triggered a safety filter or content moderation rule."
