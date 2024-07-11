from flask import Flask, request, render_template, send_file
from scipy.stats import norm
import pandas as pd
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_sample', methods=['POST'])
def calculate_sample():
    confidence = float(request.form['confidence'])
    margin = float(request.form['margin'])
    file = request.files['file']
    
    if not file:
        return "No file uploaded", 400
    
    df = pd.read_csv(file)
    population_size = len(df)
    
    z_score = norm.ppf(1 - (1 - confidence) / 2)
    p = 0.5
    numerator = (z_score ** 2) * p * (1 - p)
    denominator = margin ** 2
    sample_size = round((numerator / denominator) / (1 + ((numerator / denominator) - 1) / population_size))
    
    sampled_df = df.sample(n=sample_size)
    
    output = io.BytesIO()
    sampled_df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='sampled_data.csv')

if __name__ == '__main__':
    app.run(debug=True)
