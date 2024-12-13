from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartForConditionalGeneration, BartTokenizer
import torch
import warnings

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load GPT-2 model and tokenizer for interview questions
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token  # Set the padding token

# Load BART model and tokenizer for summarization
tokenizer_bart = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to summarize text using BART
def summarize_text(text, max_length=150):
    inputs = tokenizer_bart.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_bart.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to generate interview questions using GPT-2
def generate_interview_questions(resume_text, num_questions=3, max_length=1024):
    prompt = f"Generate {num_questions} interview questions based on the following resume:\n\n{resume_text}\n\nInterview Questions:"
    inputs = tokenizer_gpt2.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    attention_mask = torch.ones_like(inputs)
    outputs = model_gpt2.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer_gpt2.pad_token_id
    )
    generated_text = tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)
    start_index = generated_text.lower().find("interview questions:") + len("interview questions:")
    questions_text = generated_text[start_index:].strip()
    questions = questions_text.split("\n")
    questions = [q.strip() for q in questions if q.strip()]
    return "\n".join(questions[:num_questions])

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')  # Create an HTML file for file upload

@app.route('/process', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extract text from the PDF
        resume_text = extract_text_from_pdf(file_path)
        if not resume_text.strip():
            return jsonify({"error": "The resume text is empty. Please check the PDF content."}), 400
        
        # Summarize and generate interview questions
        summarized_text = summarize_text(resume_text, max_length=200)
        generated_questions = generate_interview_questions(summarized_text, num_questions=3)
        
        return jsonify({
            "summarized_resume": summarized_text,
            "interview_questions": generated_questions
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
