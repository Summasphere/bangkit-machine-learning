import os
import urllib.request
import tensorflow as tf
from ..services.gemini_llm import GeminiLLM
from ..utils.helpers import process_url

################### BART ############################
URL = (
    "https://huggingface.co/mnabielap/bart-multinews/resolve/main/bart-multinews.keras"
)
LOCAL_PATH = "bart-multinews.keras"

class BartSummarizer:  # /summarize/bart
    def __init__(self, max_length=256):
        self.download_model()
        self.max_length = max_length
        self.bart_model = tf.keras.models.load_model(
            LOCAL_PATH, 
            # custom_objects={"BartSeq2SeqLM": keras_nlp.models.BartSeq2SeqLM}
        )

    def download_model(self):
        if not os.path.exists(LOCAL_PATH):
            print(f"File {LOCAL_PATH} not found locally. Downloading from {URL}...")
            urllib.request.urlretrieve(URL, LOCAL_PATH)
            print(f"Downloaded {LOCAL_PATH} successfully.")
        else:
            print(f"File {LOCAL_PATH} already exists locally. No need to download.")

    def summarize(self, input_text):
        output = self.bart_model.generate(input_text, max_length=self.max_length)
        return output


################### GEMINI ############################
class GeminiSummarizer(GeminiLLM):
    def __init__(self, configs_path="configs/config.yaml"):
        super().__init__(configs_path)
        self.maximum_try = 10

    def process_text(self, input_text):
        system_instruction = 'Objective:\nYou are summarization application specifically designed for researchers to efficiently extract key information from academic papers. Researchers often face the challenge of sifting through numerous papers to find inspiration and relevant information, but only about 20% of a paper typically contains the critical insights they need. This application aims to expedite the literature review process by providing concise, targeted summaries focusing on the most valuable parts of each paper.\n\nKey Requirements:\n\nMethodology Summary: Clearly outline the research methods used, including experimental design, data collection, and analysis techniques.\n\nEquations: Highlight and extract every important equation if exists. The equations must be written in LaTeX so it can be rendered in markdown media. Do not let more than three equations on the same line, if there are more than three, put it in the new line. Make sure to always use the equation environment to write an equation that is given in a line, e.g. $$ H_ {k_p}=frac {Y_ {k_p}} {X_ {k_p}} $$. Also, make sure to be careful on writing equations from documents provided by user, because sometimes PDF breaks the latex format and you might write it wrong. Make sure to not forget every detail, for example you must write it like\n\n$$ A_{dot} = \\text{softmax} (  \n\\frac {QK^T} {\\sqrt{d_{model}}}  \n) V. $$  \n$$ A_{mem} = \\frac{σ(Q)M_{s-1}}\n{σ(Q)z_{s−1}} . $$  \n$$ M_{s} ← M_{s−1} + σ(K)^TV \\text{ and } z_{s} ← z_{s−1} + \\sum_{t=1}^{N} \nσ(K_{t}). $$  \n$$ M_{s} ← M_{s−1} + σ(K)^T(V − \\frac{σ(K)M_{s−1}}{σ(K)z_{s−1}}). $$\n\nMake sure to do deeper reasoning to implement the equation correctly, I know you render the equations from PDF directly, but use your knowledge to figure out how is it supposed to be written correctly. Do not just write what you saw directly.\n\nResults Summary: Highlight the main findings and outcomes of the research, emphasizing significant results and conclusions.\nCitations for Each Argument: Provide citations AND paper reference in APA style in the end of the summary, for key arguments and claims made within the paper to facilitate further reading and verification. The citation must be written in APA style, extract from the Reference Section in the input document if exists. REMEMBER, just provide the some needed citations and reference only, no need to provide all of the citations used on the paper.\n\nImportant Aspects of the Method: Identify and summarize critical aspects and innovations of the methodology that contribute to the research field.\n\nApplication Expectations:\n\nNon-Generic Summaries: The application should avoid general summaries (such as abstracts) and focus on specific sections that contain essential details for researchers.\nEfficiency and Accuracy: Ensure that the summarization process is fast and accurate, enabling researchers to quickly grasp the core contributions of each paper.\nUser-Centric Design: Tailor the application interface and features to meet the needs of researchers, allowing them to customize the type and depth of summaries they receive.\nOutcome:\nBy using you, researchers should be able to significantly reduce the time spent on literature reviews, thereby enhancing their productivity and enabling them to produce more research papers. The application should act as a valuable tool in accelerating the research process and improving the overall quality of academic work.\n\nYou are not allowed to answer another question aside summarization task. Expected responses:\n\nExample 1:\nUsers: "Hello"\nYou: <No response>\n\nExample 2:\nUsers: "Umm"\nYou: <No response>\n'
        response_text = self.generate_result(input_text, system_instruction)
        return response_text

    def run_gemini_summarizer(self, input_text, mode="text"):
        if mode == "pdf":
            input_text = self.extract_text_from_pdf_buffer(input_text)
        elif mode == "link":
            input_text = process_url(input_text)

        for _ in range(self.maximum_try):
            try:
                summary = self.process_text(input_text)
                return summary
            except Exception as e:
                print(f"error: {e}")
        return "Failed to summarize the text. Please try again later."
