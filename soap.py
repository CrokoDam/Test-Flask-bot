import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

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
        # Convert the prompt to lowercase for case-insensitive matching
        prompt = prompt.lower()

        # Check for basic greetings
        if any(greeting in prompt for greeting in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
            return "Hello! How can I assist you today?"

        # Split the data into individual guidelines
        guidelines = re.split(r'\n\s*\n', self.data)

        # Analyze the prompt to identify key topics or keywords
        topics = prompt.split()

        # Extract sentence embeddings for each guideline
        guideline_embeddings = self.model.encode(guidelines)

        # Compute the similarity score between the prompt and each guideline
        similarity_scores = util.pytorch_cos_sim(self.model.encode([prompt]), guideline_embeddings)[0]

        # Sort the guidelines based on similarity score
        sorted_indices = similarity_scores.argsort(descending=True)

        # Select the top-ranked guidelines based on similarity score and topic relevance
        selected_guidelines = []
        max_sentences = 3  # Maximum number of sentences to include in the response
        for idx in sorted_indices:
            guideline_sentences = sent_tokenize(guidelines[idx])
            for sentence in guideline_sentences:
                if any(topic in sentence.lower() for topic in topics) and len(selected_guidelines) < max_sentences:
                    selected_guidelines.append(sentence)
                elif len(selected_guidelines) >= max_sentences:
                    break
        
        # If no relevant information found, return a default response
        if not selected_guidelines:
            return "I'm sorry, I couldn't find relevant information for your question."

        return ' '.join(selected_guidelines)

    def answer_question(self, prompt):
        # Process the prompt
        answer = self.process_question(prompt)
        return answer

    def run(self):
        print("SOAP: Hello! I'm SOAP (System Operated Assistance Provider)")
        previous_answer = None
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() == 'exit':
                print("SOAP: Goodbye!")
                break
            answer = self.answer_question(prompt)
            # Ensure the bot provides a unique answer
            if answer != previous_answer:
                print("SOAP:", answer)
                previous_answer = answer
            else:
                print("SOAP: I already provided an answer related to your question.")

if __name__ == "__main__":
    bot = SOAPBot(r"C:\Users\samxali\Desktop\Bot Trail\SOAPprog\SOAP\data.txt")  # Use raw string or double backslashes
    bot.run()