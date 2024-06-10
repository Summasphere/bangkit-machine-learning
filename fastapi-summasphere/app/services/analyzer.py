import base64
import re
import string
from collections import Counter
from io import BytesIO

import google.generativeai as genai
import matplotlib.pyplot as plt
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from ..utils.helpers import process_url, sanitize_text, string_to_object

POINTER = 0


class TopicModelling:
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
        global POINTER
        if self.api_key:
            pair_api_key = self.api_key[POINTER]
            POINTER = (POINTER + 1) % len(self.api_key)  # move to the next
            api_key, email_name = pair_api_key
            print(f"Using API Key from -> {email_name}")
            return api_key
        else:
            return "No more API keys available."

    def process_text(self, input_text, mode="text"):
        if mode == "pdf":
            input_text = sanitize_text(
                input_text.decode("latin1")
            )  # Adjust decoding as necessary
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

    def run_analysis(self, text, mode="pdf", media="frontend"):
        if mode == "pdf":
            text = self.extract_text_from_pdf_buffer(text)
        elif mode == "link":
            text = process_url(text)
        topic_dist = self.process_text(text)
        topic_dist = string_to_object(
            topic_dist[topic_dist.index("[") : topic_dist.rindex("]") + 1]
        )
        wordcloud_dict = self.wordcloud(text)

        if media == "android":
            topic_dist_img = self.barplot_to_base64(topic_dist)
            wordcloud_img = self.wordcloud_to_base64(wordcloud_dict)
            dict_analysis = {
                "topic_distribution": topic_dist_img,
                "wordcloud": wordcloud_img,
            }
        else:
            dict_analysis = {
                "topic_distribution": topic_dist,
                "wordcloud": wordcloud_dict,
            }

        return dict_analysis

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

    def wordcloud(self, text):
        text = str(text).lower()
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        words = [word for word in words if len(word) > 2]

        additional_stopwords = {
            "could",
            "would",
            "never",
            "one",
            "even",
            "like",
            "said",
            "say",
            "also",
            "might",
            "must",
            "every",
            "much",
            "may",
            "two",
            "know",
            "upon",
            "without",
            "go",
            "went",
            "got",
            "put",
            "see",
            "seem",
            "seemed",
            "take",
            "taken",
            "make",
            "made",
            "come",
            "came",
            "look",
            "looking",
            "think",
            "thinking",
            "thought",
            "use",
            "used",
            "find",
            "found",
            "give",
            "given",
            "tell",
            "told",
            "ask",
            "asked",
            "back",
            "get",
            "getting",
            "keep",
            "kept",
            "let",
            "lets",
            "seems",
            "leave",
            "left",
            "set",
            "from",
            "subject",
            "re",
            "edu",
            "use",
        }
        custom_stopwords = set(stopwords.words("english")).union(additional_stopwords)

        filtered_words = [word for word in words if word not in custom_stopwords]

        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        word_freq = Counter(lemmatized_words)
        sorted_word_freq = sorted(
            word_freq.items(), key=lambda item: item[1], reverse=True
        )
        temp_dict = dict()
        for key, val in sorted_word_freq:
            temp_dict[key] = val
        return temp_dict

    def barplot_to_base64(self, data):
        # Extracting data for plotting
        topics = [item["topic"] for item in data]
        percentages = [item["percentage"] for item in data]

        # Creating the barplot
        plt.figure(figsize=(10, 6))
        plt.barh(topics, percentages, color="skyblue")
        plt.xlabel("Percentage")
        plt.ylabel("Topic")
        plt.title("Distribution of Topics")

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Encode the BytesIO object to base64
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return img_base64

    def wordcloud_to_base64(self, word_freq):
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode("utf-8")
        return img_str
