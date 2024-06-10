from io import BytesIO

import google.generativeai as genai
import yaml
from PyPDF2 import PdfReader

from ..utils.helpers import sanitize_text

POINTER = 0  # Define POINTER globally


class GeminiLLM:
    def __init__(self, configs_path="configs/config.yaml"):
        with open(configs_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.api_key = self.config["GEMINI_API_KEY_COLLECTION"]
        self.generation_conf = self.config["generation_config"]
        self.safety_settings = [
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

    def pick_random_key(self):
        global POINTER  # Use the global POINTER variable
        if self.api_key:
            pair_api_key = self.api_key[POINTER]
            POINTER = (POINTER + 1) % len(self.api_key)  # move to the next
            api_key, email_name = pair_api_key
            print(f"Using API Key from -> {email_name}")
            return api_key
        else:
            return "No more API keys available."

    def generate_result(self, input_text, system_instruction):
        input_text = sanitize_text(input_text)
        api_key = self.pick_random_key()
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=self.safety_settings,
            generation_config=self.generation_conf,
            system_instruction=system_instruction,
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)

        response_text = response.text
        return response_text

    def extract_text_from_pdf_buffer(self, pdf_buffer):
        if isinstance(pdf_buffer, bytes):
            pdf_buffer = BytesIO(pdf_buffer)  # Convert bytes to BytesIO

        pdf_reader = PdfReader(pdf_buffer)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text = sanitize_text(text)
        return text

    def create_str_json_example(self, num_topic: int):
        res = "[\n"
        for no_topic in range(num_topic):
            no_topic += 1
            comma = "," if no_topic != num_topic else ""
            temp = (
                """
            {
                "topic":"{name_topic_""".strip()
                + str(no_topic)
                + """}",
                "percentage":{percentage_""".strip()
                + str(no_topic)
                + """},
                "detail":"{explanation_""".strip()
                + str(no_topic)
                + """}"
            }
            """.strip()
                + str(comma)
            )
            res += f"  {temp}\n"
        res += "]"
        return res
