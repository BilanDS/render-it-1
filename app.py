import os
import numpy as np
from flask import Flask, request, render_template_string
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

app = Flask(__name__)

database_url = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

model_path = 'final_skin_model.h5'
model = None

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully")
    else:
        print("Model file not found")
except Exception as e:
    print(f"Error loading model: {e}")

CLASSES = {
    0: 'AKIEC - Actinic keratoses',
    1: 'BCC - Basal cell carcinoma',
    2: 'BKL - Benign keratosis',
    3: 'DF - Dermatofibroma',
    4: 'MEL - Melanoma',
    5: 'NV - Melanocytic nevi',
    6: 'VASC - Vascular lesions'
}

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    analyses = db.relationship('Analysis', backref='patient', lazy=True)

class BodyPart(db.Model):
    __tablename__ = 'body_parts'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    analyses = db.relationship('Analysis', backref='location', lazy=True)

class DiseaseInfo(db.Model):
    __tablename__ = 'diseases'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    severity = db.Column(db.Integer, default=1)
    analyses = db.relationship('Analysis', backref='disease', lazy=True)
    recommendations = db.relationship('Recommendation', backref='disease', lazy=True)

class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=False)

class Analysis(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    body_part_id = db.Column(db.Integer, db.ForeignKey('body_parts.id'), nullable=True)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=True)

with app.app_context():
    db.create_all()

HOME_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Derma Lab</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 20px auto; padding: 20px; background: #f9f9f9; }
        .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 15px; }
        label { display: block; font-weight: bold; margin-bottom: 5px; }
        input, select, button { width: 100%; padding: 10px; box-sizing: border-box; }
        button { background: #28a745; color: white; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #218838; }
        .link { display: block; text-align: center; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center;">Skin Diagnosis</h1>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label>Patient Name:</label>
                <input type="text" name="username" required placeholder="Name">
            </div>
            <div class="form-group">
                <label>Location:</label>
                <select name="body_part">
                    <option value="Face">Face</option>
                    <option value="Arm">Arm</option>
                    <option value="Back">Back</option>
                    <option value="Leg">Leg</option>
                    <option value="Other">Other</option>
                </select>
            </div>
            <div class="form-group">
                <label>Photo:</label>
                <input type="file" name="file" accept="image/*" required>
            </div>
            <button type="submit">Analyze</button>
        </form>
        <div class="link">
            <a href="/view-data">View History</a> | 
            <a href="/reset-db" style="color: red; font-size: 0.8em;">Reset DB</a>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None:
        return "<h3>Error: Model not loaded.</h3>"

    username = request.form.get('username')
    body_part_name = request.form.get('body_part')
    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        preds = model.predict(x)
        idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]) * 100)
        
        full_text = CLASSES[idx]
        short_name = full_text.split(' - ')[0]

        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username)
            db.session.add(user)
        
        bp = BodyPart.query.filter_by(name=body_part_name).first()
        if not bp:
            bp = BodyPart(name=body_part_name)
            db.session.add(bp)
            db.session.commit()
            
        disease = DiseaseInfo.query.filter_by(name=short_name).first()
        
        new_analysis = Analysis(
            image_name=file.filename,
            confidence=confidence,
            patient=user,
            location=bp,
            disease=disease
        )
        
        db.session.add(new_analysis)
        db.session.commit()

        color = "red" if disease and disease.severity > 5 else "green"
        recs_html = ""
        if disease:
            for r in disease.recommendations:
                recs_html += f"<li>{r.text}</li>"

        return f'''
        <div style="font-family: sans-serif; max-width: 600px; margin: 20px auto; padding: 20px; border: 1px solid #ccc; border-radius: 8px;">
            <h1 style="color: {color};">Result: {full_text}</h1>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            <p><strong>Patient:</strong> {username}</p>
            <hr>
            <h3>Recommendations:</h3>
            <ul>{recs_html}</ul>
            <br>
            <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">Back</a>
        </div>
        '''

    except Exception as e:
        return f"Error processing image: {str(e)}"

@app.route('/view-data')
def view_data():
    analyses = Analysis.query.order_by(Analysis.timestamp.desc()).all()
    
    html = '''
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 20px auto; }
        .card { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .danger { color: red; font-weight: bold; }
        .safe { color: green; font-weight: bold; }
    </style>
    <h1>Diagnosis History</h1>
    <a href="/">Back to Home</a><br><br>
    '''
    
    for a in analyses:
        if a.disease:
            d_name = a.disease.description 
            severity_color = "danger" if a.disease.severity > 5 else "safe"
            recs = "".join([f"<li>{r.text}</li>" for r in a.disease.recommendations])
        else:
            d_name = "Unknown"
            severity_color = "black"
            recs = "<li>No data</li>"

        html += f'''
        <div class="card">
            <h3>Image: {a.image_name}</h3>
            <p><strong>Patient:</strong> {a.patient.username}</p>
            <p><strong>Location:</strong> {a.location.name if a.location else 'Not specified'}</p>
            <hr>
            <p><strong>Result:</strong> <span class="{severity_color}">{d_name}</span></p>
            <p><strong>Confidence:</strong> {a.confidence:.2f}%</p>
            <p><strong>Recommendations:</strong></p>
            <ul>{recs}</ul>
            <small>Date: {a.timestamp.strftime('%Y-%m-%d %H:%M')}</small>
        </div>
        '''
    
    return html

@app.route('/reset-db')
def reset_db():
    try:
        db.drop_all()
        db.create_all()

        parts = ["Face", "Arm", "Back", "Leg", "Other"]
        for p in parts:
            db.session.add(BodyPart(name=p))
        
        for idx, text in CLASSES.items():
            short = text.split(' - ')[0]
            
            if short in ['MEL', 'BCC', 'AKIEC', 'VASC']:
                sev = 10
                rec_text = "Urgent consultation required!"
            else:
                sev = 1
                rec_text = "Benign. Observe for changes."

            d = DiseaseInfo(name=short, description=text, severity=sev)
            db.session.add(d)
            
            rec = Recommendation(text=rec_text, disease=d)
            db.session.add(rec)

        db.session.commit()
        return "Database reset and seeded successfully."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
