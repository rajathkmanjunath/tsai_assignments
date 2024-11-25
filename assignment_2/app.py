# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle animal selection
@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.json
    animal = data.get('animal')
    if animal in ['cat', 'dog', 'elephant']:
        return jsonify({'image': f"/static/{animal}.jpg"})
    return jsonify({'error': 'Invalid animal selected'}), 400

# Route to handle file upload
@app.route('/upload_file', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        file_info = {
            'name': uploaded_file.filename,
            'size': len(uploaded_file.read()),
            'type': uploaded_file.content_type
        }
        return jsonify(file_info)
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)
