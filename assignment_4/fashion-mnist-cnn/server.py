from flask import Flask, render_template, jsonify
import os

# Update these paths to be relative to the server.py location
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

app = Flask(__name__,
            template_folder=template_dir,
            static_folder=static_dir)

@app.route('/')
def index():
    return render_template('monitor.html')

@app.route('/get_training_state')
def get_training_state():
    try:
        with open(os.path.join(static_dir, 'training_state.json'), 'r') as f:
            return jsonify(f.read())
    except:
        return jsonify({'epoch': 0, 'loss': 0, 'accuracy': 0})

@app.route('/get_results')
def get_results():
    try:
        with open(os.path.join(static_dir, 'results.txt'), 'r') as f:
            return jsonify({'image': f.read()})
    except:
        return jsonify({'image': ''})

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    app.run(debug=True) 