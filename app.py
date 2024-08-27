from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from io import BytesIO
import torch


app = Flask(__name__)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")

@app.route('/',methods=['GET'])
def home():
    return "Flask server is running!"


@app.route('/ai', methods=['POST'])
def ai():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    question = request.form.get('question', '')
    if question == '':
        return jsonify({'error': 'No question provided'}), 400

    try:
        image = Image.open(BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
    try:
        inputs = processor(image, question, return_tensors="pt")
        with torch.no_grad():
            generated_ids = model.generate(**inputs)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        return jsonify({'question': question, 'answer': generated_text})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True,port=8001,use_reloader=False)