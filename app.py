from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

import gwa_scanner
import id_scanner
import city_scanner

from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
import json

app = Flask(__name__, template_folder='.', static_folder='css')
CORS(app)


UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

spell = SpellChecker()

def correct_sentence(sentence):
    corrected = [spell.correction(word) or word for word in sentence.split()]
    return ' '.join(corrected)

model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    with open('chatbot_data.json', 'r') as f:
        faq_list = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    print("Error loading chatbot_data.json.")
    faq_list = []

faq_questions = [item['question'] for item in faq_list]
faq_answers = [item['answer'] for item in faq_list]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

@app.route('/')
def index():
    return render_template('home_page.html')

@app.route('/admissions/uploads', methods=['POST'])
def upload_file():
    try:
        if 'school-record-upload' in request.files:
            file = request.files['school-record-upload']
            filename = secure_filename(file.filename)
            user_id = request.form.get('user_id')

            if user_id:
                ext = os.path.splitext(filename)[1]
                filename = f"{secure_filename(user_id)}{ext}"

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_gwa = gwa_scanner.extract_gwa(filepath)

            if 'gwa' in extracted_gwa:
                return jsonify({"status": "success", "gwa": extracted_gwa['gwa']}), 200
            else:
                return jsonify({"status": "error", "error": "GWA extraction failed"}), 500

        elif 'exam-result-upload' in request.files:
            file = request.files['exam-result-upload']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_exam_info = id_scanner.extract_exam_info_from_image(filepath)

            if extracted_exam_info:
                return jsonify({"status": "success", "exam_info": extracted_exam_info}), 200
            else:
                return jsonify({"status": "error", "error": "Exam information extraction failed"}), 500

        elif 'voters-id-upload' in request.files:
            file = request.files['voters-id-upload']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_city_info = city_scanner.extract_city_info(filepath)

            if extracted_city_info:
                extracted_city_info['id_type'] = "Voter's ID"
                return jsonify({"status": "success", "voters_info": extracted_city_info}), 200
            else:
                return jsonify({"status": "error", "error": "City information extraction failed"}), 500

        else:
            return jsonify({"status": "error", "error": "No valid file uploaded"}), 400

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question', '')
    if not user_input:
        return jsonify({'answer': "Please ask a question about admissions."})

    user_input_corrected = correct_sentence(user_input)
    user_embedding = model.encode(user_input_corrected, convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, faq_embeddings)[0]
    best_match_idx = scores.argmax().item()
    best_score = scores[best_match_idx].item()

    similarity_threshold = 0.7  

    if best_score < similarity_threshold:
        return jsonify({'answer': "I'm sorry, I couldn't understand your question. Could you please ask something else?"})

    return jsonify({'answer': faq_answers[best_match_idx]})

@app.route('/get_faqs', methods=['GET'])
def get_faqs():
    return jsonify(faq_list), 200

@app.route('/add_faq', methods=['POST'])
def add_faq():
    try:
        data = request.get_json()
        new_question = data.get('question')
        new_answer = data.get('answer')

        if not new_question or not new_answer:
            return jsonify({"error": "Both question and answer are required."}), 400

        faq_id = len(faq_list) + 1
        faq_list.append({'id': faq_id, 'question': new_question, 'answer': new_answer})
        faq_questions.append(new_question)
        faq_answers.append(new_answer)
        global faq_embeddings
        faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

        with open('chatbot_data.json', 'w') as f:
            json.dump(faq_list, f)

        return jsonify({"success": True, "message": "Question added successfully!"}), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/delete_faq/<int:faq_id>', methods=['DELETE'])
def delete_faq(faq_id):
    try:
        global faq_list
        faq_list = [faq for faq in faq_list if faq['id'] != faq_id]

        faq_questions = [item['question'] for item in faq_list]
        faq_answers = [item['answer'] for item in faq_list]
        global faq_embeddings
        faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

        with open('chatbot_data.json', 'w') as f:
            json.dump(faq_list, f)

        return jsonify({"success": True, "message": "FAQ deleted successfully!"}), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
