import os
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

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    analyses = db.relationship('Analysis', backref='author', lazy=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(100), nullable=False)  # Назва файлу фото
    prediction = db.Column(db.String(100), nullable=False)  # Результат (діагноз)
    confidence = db.Column(db.Float, nullable=False)        # Точність (0.0 - 1.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) # Час аналізу
    
    # Прив'язка до користувача
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return "Сервіс діагностики шкіри працює! База даних підключена."

@app.route('/test-db')
def test_db():
    try:
        user = User.query.filter_by(email="test@example.com").first()
        if not user:
            user = User(username="TestUser", email="test@example.com")
            db.session.add(user)
            db.session.commit()
        
        fake_analysis = Analysis(
            image_name="mole_sample_01.jpg",
            prediction="Benign Nevus (Родимка)",
            confidence=0.98,
            author=user
        )
        db.session.add(fake_analysis)
        db.session.commit()
        
        return f"Успіх! Записано в БД: Юзер {user.username}, Діагноз: {fake_analysis.prediction}"
    except Exception as e:
        return f"Помилка при роботі з БД: {str(e)}"

@app.route('/view-data')
def view_data():

    all_records = Analysis.query.all()
    
    html_response = "<h1>Історія діагностики</h1><ul>"
    
    for record in all_records:
        html_response += f"""
            <li>
                <strong>ID:</strong> {record.id}<br>
                <strong>Користувач:</strong> {record.author.username} ({record.author.email})<br>
                <strong>Файл:</strong> {record.image_name}<br>
                <strong>Результат:</strong> {record.prediction} (Впевненість: {record.confidence})<br>
                <strong>Дата:</strong> {record.timestamp}
                <hr>
            </li>
        """
    html_response += "</ul>"
    return html_response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
