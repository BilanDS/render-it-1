# AI Derma Lab: Інтелектуальна система діагностики шкіри

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)
![Render](https://img.shields.io/badge/Deployment-Render-purple)
![CI Status](https://github.com/ВАШ_НІК/ВАШ_РЕПОЗИТОРІЙ/actions/workflows/ci_test.yml/badge.svg)

Веб-застосунок для автоматизованої діагностики дерматологічних захворювань за допомогою згорткової нейронної мережі (CNN). Система дозволяє користувачам завантажувати зображення уражень шкіри та отримувати миттєву класифікацію ризиків із рекомендаціями.

---

## Основний функціонал

1.  **AI-Діагностика:**
    * Використання оптимізованої моделі **TensorFlow Lite** (MobileNetV2 based) для швидкої класифікації.
    * Розпізнавання **7 класів** шкірних утворень.
    * Визначення рівня впевненості (Confidence Score).

2.  **Управління даними:**
    * Збереження результатів аналізу в хмарній базі даних **PostgreSQL**.
    * Збереження зображень пацієнтів безпосередньо в БД (BLOB format).
    * Автоматична генерація рекомендацій залежно від діагнозу.

3.  **Інтерфейс:**
    * Попередній перегляд (Preview) зображення перед завантаженням.
    * Історія діагностики з візуалізацією збережених знімків.
    * Адаптивний веб-дизайн.

4.  **DevOps & CI/CD:**
    * Автоматичне тестування коду через **GitHub Actions**.
    * Автоматичне розгортання (Deploy) на платформі **Render**.

---

## Класифікація хвороб (Модель)

Модель навчена розпізнавати такі типи уражень:
* **nv** — Меланоцитарний невус (Родимка)
* **mel** — Меланома (Злоякісне)
* **bkl** — Доброякісний кератоз
* **bcc** — Базальноклітинна карцинома
* **akiec** — Актинічний кератоз
* **vasc** — Судинні ураження
* **df** — Дерматофіброма

---

## Технологічний стек

* **Мова програмування:** Python 3.11
* **Web Framework:** Flask
* **Machine Learning:** TensorFlow, Keras, TFLite Runtime
* **База даних:** PostgreSQL (SQLAlchemy ORM)
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
* **Хостинг:** Render Cloud

---

## Як запустити локально

Щоб запустити проєкт на своєму комп'ютері для розробки:

1.  **Клонуйте репозиторій:**
    ```bash
    git clone [https://github.com/ВАШ_НІК/ВАШ_РЕПОЗИТОРІЙ.git](https://github.com/ВАШ_НІК/ВАШ_РЕПОЗИТОРІЙ.git)
    cd ВАШ_РЕПОЗИТОРІЙ
    ```

2.  **Створіть віртуальне середовище:**
    ```bash
    python -m venv venv
    # Для Windows:
    venv\Scripts\activate
    # Для Mac/Linux:
    source venv/bin/activate
    ```

3.  **Встановіть залежності:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Запустіть сервер:**
    ```bash
    python app.py
    ```

5.  **Відкрийте у браузері:**
    Перейдіть за адресою `http://127.0.0.1:10000`

---

##  Структура проєкту

```text
├── .github/workflows/   # Налаштування CI/CD (GitHub Actions)
├── app.py               # Головний файл застосунку (Backend + Frontend)
├── skin_model.tflite    # Навчена модель нейромережі
├── requirements.txt     # Список бібліотек
├── Procfile             # Інструкція запуску для Render
└── README.md            # Документація
