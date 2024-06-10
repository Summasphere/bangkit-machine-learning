from io import BytesIO

import google.generativeai as genai
import yaml

from ..utils.helpers import process_url, sanitize_text

POINTER = 0  # Define POINTER globally


class GeminiSummarizer:
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

    def extract_text_from_pdf_buffer(self, pdf_buffer):
        from PyPDF2 import PdfReader

        if isinstance(pdf_buffer, bytes):
            pdf_buffer = BytesIO(pdf_buffer)  # Convert bytes to BytesIO

        pdf_reader = PdfReader(pdf_buffer)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        text = sanitize_text(text)
        return text

    def process_text(self, input_text, mode="text"):
        if mode == "pdf":
            input_text = self.extract_text_from_pdf_buffer(input_text)
        elif mode == "link":
            input_text = process_url(input_text)

        print(input_text)

        api_key = self.pick_random_key()
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=self.safety_settings,
            generation_config=self.generation_conf,
            system_instruction="Objective: Analyze the 5 most related topics in the text below. Explain each topic completely. Explain why the text fits the topic. Provide in JSON format.",
        )

        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(input_text)

        response_text = response.text
        return response_text
