import os
import numpy as np
from flask import Flask, request, render_template_string
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

database_url = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)

model_path = 'skin_model.tflite'
interpreter = None
input_details = None
output_details = None

try:
    if os.path.exists(model_path):
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded")
    else:
        print("Model file not found")
except Exception as e:
    print(f"Error: {e}")

CLASSES = {
    0: 'AKIEC - Актинічний кератоз',
    1: 'BCC - Базаліома',
    2: 'BKL - Доброякісний кератоз',
    3: 'DF - Дерматофіброма',
    4: 'MEL - Меланома',
    5: 'NV - Невус (Родимка)',
    6: 'VASC - Судинні ураження'
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
    <title>AI Дерматологія</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
        <h1 style="text-align: center;">Діагностика шкіри</h1>
        <form action="/analyze" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label>Ім'я пацієнта:</label>
                <input type="text" name="username" required placeholder="Введіть ім'я">
            </div>
            <div class="form-group">
                <label>Локалізація:</label>
                <select name="body_part">
                    <option value="Обличчя">Обличчя</option>
                    <option value="Рука">Рука</option>
                    <option value="Спина">Спина</option>
                    <option value="Нога">Нога</option>
                    <option value="Інше">Інше</option>
                </select>
            </div>
            <div class="form-group">
                <label>Фото:</label>
                <input type="file" name="file" accept="image/*" required>
            </div>
            <button type="submit">Аналізувати</button>
        </form>
        <div class="link">
            <a href="/view-data">Історія аналізів</a> | 
            <a href="/reset-db" style="color: red; font-size: 0.8em;">Оновити БД</a>
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
    if interpreter is None:
        return "<h3>Помилка: Модель не завантажена.</h3>"

    username = request.form.get('username')
    body_part_name = request.form.get('body_part')
    file = request.files['file']

    if file.filename == '':
        return "Файл не обрано"

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])

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
            <h1 style="color: {color};">Результат: {full_text}</h1>
            <p><strong>Впевненість:</strong> {confidence:.2f}%</p>
            <p><strong>Пацієнт:</strong> {username}</p>
            <hr>
            <h3>Рекомендації:</h3>
            <ul>{recs_html}</ul>
            <br>
            <a href="/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px;">Назад</a>
        </div>
        '''

    except Exception as e:
        return f"Помилка обробки: {str(e)}"

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
    <h1>Історія діагностики</h1>
    <a href="/">На головну</a><br><br>
    '''
    
    for a in analyses:
        if a.disease:
            d_name = a.disease.description 
            severity_color = "danger" if a.disease.severity > 5 else "safe"
            recs = "".join([f"<li>{r.text}</li>" for r in a.disease.recommendations])
        else:
            d_name = "Невідомо"
            severity_color = "black"
            recs = "<li>Немає даних</li>"

        html += f'''
        <div class="card">
            <h3>Зображення: {a.image_name}</h3>
            <p><strong>Пацієнт:</strong> {a.patient.username}</p>
            <p><strong>Локалізація:</strong> {a.location.name if a.location else 'Не вказано'}</p>
            <hr>
            <p><strong>Результат:</strong> <span class="{severity_color}">{d_name}</span></p>
            <p><strong>Впевненість:</strong> {a.confidence:.2f}%</p>
            <p><strong>Рекомендації:</strong></p>
            <ul>{recs}</ul>
            <small>Дата: {a.timestamp.strftime('%Y-%m-%d %H:%M')}</small>
        </div>
        '''
    
    return html

@app.route('/reset-db')
def reset_db():
    try:
        db.drop_all()
        db.create_all()

        parts = ["Обличчя", "Рука", "Спина", "Нога", "Інше"]
        for p in parts:
            db.session.add(BodyPart(name=p))
        
        for idx, text in CLASSES.items():
            short = text.split(' - ')[0]
            
            if short in ['MEL', 'BCC', 'AKIEC', 'VASC']:
                sev = 10
                rec_text = "Термінова консультація лікаря!"
            else:
                sev = 1
                rec_text = "Доброякісне. Спостерігайте за змінами."

            d = DiseaseInfo(name=short, description=text, severity=sev)
            db.session.add(d)
            
            rec = Recommendation(text=rec_text, disease=d)
            db.session.add(rec)

        db.session.commit()
        return "База даних успішно оновлена."
    except Exception as e:
        return f"Помилка: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
