import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from flask import Flask, render_template, request, jsonify

# install pip directories flask sentence-transformers requests transformers nltk python.exe -m pip install --upgrade pip

app = Flask(__name__, template_folder='C:/Users/samxali/Desktop/Bot Trail/SOAPprog/SOAP/templates')

class SOAPBot:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:
            data = file.read()
        return data

    def process_question(self, prompt):
        prompt = prompt.lower()

        if any(greeting in prompt for greeting in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
            return "Hello! How can I assist you today?"
        
        if any(greeting in prompt for greeting in ["how are you", "how is it going", "wassup", "how r u", "how you doing today?"]):
            return "I'm doing good, thanks for asking. How may I assist you with?"

        guidelines = re.split(r'\n\s*\n', self.data)

        topics = prompt.split()

        guideline_embeddings = self.model.encode(guidelines)

        similarity_scores = util.pytorch_cos_sim(self.model.encode([prompt]), guideline_embeddings)[0]

        sorted_indices = similarity_scores.argsort(descending=True)

        selected_guidelines = []
        max_sentences = 3  
        for idx in sorted_indices:
            guideline_sentences = sent_tokenize(guidelines[idx])
            for sentence in guideline_sentences:
                if any(topic in sentence.lower() for topic in topics) and len(selected_guidelines) < max_sentences:
                    selected_guidelines.append(sentence)
                elif len(selected_guidelines) >= max_sentences:
                    break
        
        if not selected_guidelines:
            return "I'm sorry, I could you rephrase?"

        return ' '.join(selected_guidelines)

    def answer_question(self, prompt):
        answer = self.process_question(prompt)
        return answer

bot = SOAPBot(r"C:\Users\samxali\Desktop\Bot Trail\SOAPprog\SOAP\data.txt")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["user_input"]
    bot_response = bot.answer_question(user_input)
    return jsonify({"SOAP": bot_response})

if __name__ == "__main__":
    app.run(debug=True, port=5001)