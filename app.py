import os
import random
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

database_url = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Користувачі
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    
    analyses = db.relationship('Analysis', backref='patient', lazy=True)

# Частина тіла
class BodyPart(db.Model):
    __tablename__ = 'body_parts'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    
    analyses = db.relationship('Analysis', backref='location', lazy=True)

# Хвороби
class DiseaseInfo(db.Model):
    __tablename__ = 'diseases'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False) 
    description = db.Column(db.Text, nullable=True)               
    severity = db.Column(db.Integer, default=1)                   
    
    analyses = db.relationship('Analysis', backref='disease', lazy=True)
    recommendations = db.relationship('Recommendation', backref='disease', lazy=True)

# Рекомендації (лікування)
class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=False)

# Аналізи
class Analysis(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    body_part_id = db.Column(db.Integer, db.ForeignKey('body_parts.id'), nullable=True)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=True) # Що знайшов ШІ

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return "Медичний сервіс працює! Структура БД оновлена до 5 таблиць."

@app.route('/reset-db')
def reset_db():
    try:
        db.drop_all()
        db.create_all()

        # Заповнення БД
        parts = [
            BodyPart(name="Обличчя"), 
            BodyPart(name="Спина"), 
            BodyPart(name="Рука (передпліччя)"), 
            BodyPart(name="Нога")
        ]
        db.session.add_all(parts)
        db.session.commit()

        d1 = DiseaseInfo(name="Benign Nevus", description="Безпечна родимка. Не потребує лікування.", severity=1)
        d2 = DiseaseInfo(name="Melanoma", description="Злоякісне утворення. Потребує термінового втручання!", severity=10)
        d3 = DiseaseInfo(name="Seborrheic Keratosis", description="Доброякісне вікове утворення шкіри.", severity=2)
        
        db.session.add_all([d1, d2, d3])
        db.session.commit()

        r1 = Recommendation(text="Спостерігайте за зміною розміру раз на 6 місяців.", disease=d1)
        r2 = Recommendation(text="Терміново запишіться до онколога!", disease=d2)
        r3 = Recommendation(text="Можна видалити з косметичною метою.", disease=d3)
        
        db.session.add_all([r1, r2, r3])
        db.session.commit()

        user = User(username="Student_Lab", email="student@example.com")
        db.session.add(user)
        db.session.commit()

        a1 = Analysis(
            image_name="scan_001.jpg", 
            confidence=0.98, 
            patient=user, 
            location=parts[1],
            disease=d1
        )
        a2 = Analysis(
            image_name="scan_alert_02.png", 
            confidence=0.89, 
            patient=user, 
            location=parts[0], 
            disease=d2         
        )
        
        db.session.add_all([a1, a2])
        db.session.commit()

        return "Базу даних успішно оновлено! Створено 5 таблиць та тестові записи."
    except Exception as e:
        return f"Помилка: {str(e)}"

@app.route('/view-data')
def view_data():
    analyses = Analysis.query.all()
    
    html = """
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 20px auto; }
        .card { border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; }
        .danger { color: red; font-weight: bold; }
        .safe { color: green; }
    </style>
    <h1>Результати діагностики (Звіт з 5 таблиць)</h1>
    """
    
    for a in analyses:
        severity_color = "danger" if a.disease.severity > 5 else "safe"
        
        recs = "".join([f"<li>{r.text}</li>" for r in a.disease.recommendations])
        
        html += f"""
        <div class="card">
            <h3>Зображення: {a.image_name}</h3>
            <p><strong>Пацієнт:</strong> {a.patient.username}</p>
            <p><strong>Локалізація:</strong> {a.location.name}</p>
            <hr>
            <p><strong>Результат ШІ:</strong> <span class="{severity_color}">{a.disease.name}</span></p>
            <p><strong>Опис:</strong> {a.disease.description}</p>
            <p><strong>Впевненість:</strong> {a.confidence * 100}%</p>
            <p><strong>Рекомендації:</strong></p>
            <ul>{recs}</ul>
            <small>Дата: {a.timestamp}</small>
        </div>
        """
    
    return html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
