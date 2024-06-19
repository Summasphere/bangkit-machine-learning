import base64
import re
import string
from collections import Counter
from io import BytesIO

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from ..services.gemini_llm import GeminiLLM
from ..utils.helpers import process_url, string_to_object


class TopicModelling(GeminiLLM):
    def __init__(self, configs_path="configs/config.yaml"):
        super().__init__(configs_path)
        self.maximum_try = 10

    def process_text(self, input_text):
        system_instruction = f"""
            Analyze the 5 most related topics in the text below. explain each topic completely. explain why the text fits the topic.
            >> Constraint:
            - minimum 100 words and maximum 500 words
            - also explain how many percent of each topic is related to the text.
            - provide in a formal, no-nonsense format so that these results can be used for academic purposes.
            - to the point
            - the output must be json format for example
            {self.create_str_json_example(5)}
            - do not provide any text outside of json
            """.strip()
        response_text = self.generate_result(input_text, system_instruction)
        return response_text

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

    ######################## ANDROID UTILITY ########################
    def barplot_to_base64(self, data):
        # Extracting data for plotting
        topics = [item['topic'] for item in data]
        percentages = [item['percentage'] for item in data]

        # Creating the barplot
        plt.figure(figsize=(10, 8))
        plt.barh(topics, percentages, color='skyblue')
        plt.xlabel('Percentage')
        plt.ylabel('Topic')
        plt.title('Distribution of Topics')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest percentage on top
        
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the BytesIO object to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
    
        return img_base64

    def wordcloud_to_base64(self, word_freq):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(15, 15))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return img_str
    ########################################################################

    def run_analysis(self, text, mode="pdf", media="frontend"):
        if mode == "pdf":
            text = self.extract_text_from_pdf_buffer(text)
        elif mode == "link":
            text = process_url(text)

        for _ in range(self.maximum_try):
            try:
                topic_dist = self.process_text(text)
                break
            except Exception as e:
                print(f"error: {e}")
        try:
            topic_dist = string_to_object(
                topic_dist[topic_dist.index("[") : topic_dist.rindex("]") + 1]
            )
        except Exception:
            topic_dist = string_to_object(
                topic_dist[topic_dist.index("{") : topic_dist.rindex("}") + 1]
            )

        wordcloud_dict = self.wordcloud(text)

        if media == "android":
            topic_desc = "\n".join([f"{i+1}. " + item["topic"] + "\n\n" + item['detail'] for (i, item) in enumerate(topic_dist)])
            topic_dist_img = self.barplot_to_base64(topic_dist)
            wordcloud_img = self.wordcloud_to_base64(wordcloud_dict)
            dict_analysis = {"topic_distribution": topic_dist_img, "topic_desc": topic_desc, "wordcloud": wordcloud_img}
        else:
            dict_analysis = {
                "topic_distribution": topic_dist,
                "wordcloud": wordcloud_dict,
            }

        return dict_analysis
