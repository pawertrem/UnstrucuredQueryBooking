from flask import Flask, render_template, request, abort, session
import pandas as pd
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'c9f651cdcd7893a508869adbca17ccb9222aad0d259540c11d1a539b7deb5dd9'
app.config['SESSION_COOKIE_NAME'] = "session"
app.config['SESSION_COOKIE_SECURE'] = False

def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

def log_user_action(user_id, action):
    import sqlite3
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_logs (user_id, action) VALUES (?, ?)
    ''', (user_id, action))
    conn.commit()
    conn.close()

# Загрузка данных из JSON
def load_data():
    df = pd.read_json('final_data.json', orient='records', lines=True)
    
    # Преобразуем рейтинг: '4,99' -> '4.99', 'Новое' -> NaN
    df['rating'] = df.get('rating', '0').astype(str).str.replace(',', '.').str.strip()

    # Заменяем нечисловые значения типа 'Новое', 'None', 'N/A'
    df['rating'] = df['rating'].apply(lambda x: x if x.replace('.', '', 1).isdigit() else None)

    # Конвертируем в float и заполняем NaN = 0
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)

    df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
    df['guests'] = pd.to_numeric(df.get('guests', 0), errors='coerce').fillna(0).astype(int)
    df['bedrooms'] = pd.to_numeric(df.get('bedrooms', 0), errors='coerce').fillna(0).astype(int)
    df['beds'] = pd.to_numeric(df.get('beds', 0), errors='coerce').fillna(0).astype(int)
    df['amenities'] = df.get('amenities', [[]]).apply(lambda x: x if isinstance(x, list) else [])
    return df

def filter_properties(min_price, max_price, min_rating, sort_by,
                      min_guests=None, min_bedrooms=None, min_beds=None,
                      required_amenities=None):
    df = load_data()
    filtered = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    filtered = filtered[filtered['rating'] >= min_rating]

    if min_guests:
        filtered = filtered[filtered['guests'] >= min_guests]
    if min_bedrooms:
        filtered = filtered[filtered['bedrooms'] >= min_bedrooms]
    if min_beds:
        filtered = filtered[filtered['beds'] >= min_beds]
    if required_amenities and 'amenities' in filtered.columns:
        # Приводим к списку если что
        filtered['amenities'] = filtered['amenities'].apply(lambda x: x if isinstance(x, list) else [])

        for amenity in required_amenities:
            if len(filtered) == 0:
                break
            filtered = filtered[filtered['amenities'].apply(lambda x: amenity in x)]

    if sort_by == 'price-asc':
        filtered = filtered.sort_values('price')
    elif sort_by == 'price-desc':
        filtered = filtered.sort_values('price', ascending=False)
    elif sort_by == 'rating-desc':
        filtered = filtered.sort_values('rating', ascending=False)

    return filtered.to_dict('records')

@app.route('/ping')
def ping():
    return "Ping OK!"

@app.route('/')
def home():

    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    default_filters = {
        'min-price': 0,
        'max-price': 10000,
        'rating': 0,
        'sort-by': 'rating-desc',
        'guests': 0,
        'bedrooms': 0,
        'beds': 0,
        'amenities': []
    }
    return render_template('filter_version.html', 
                            results=None, 
                            filters=default_filters,
                            current_page=0,
                            total_pages=0)

@app.route('/search', methods=['GET', 'POST'])
def search():

    # Генерация user_id при первом заходе
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    if request.method == 'POST':
        min_price = float(request.form.get('min-price', 0))
        max_price = float(request.form.get('max-price', 10000))
        min_rating = float(request.form.get('rating', 0))
        sort_by = request.form.get('sort-by', 'rating-desc')
        page = 1

        min_guests = safe_int(request.form.get('guests'), 0)
        min_bedrooms = safe_int(request.form.get('bedrooms'), 0)
        min_beds = safe_int(request.form.get('beds'), 0)
        required_amenities = request.form.getlist('amenities')

    else:
        min_price = float(request.args.get('min-price', 0))
        max_price = float(request.args.get('max-price', 10000))
        min_rating = float(request.args.get('rating', 0))
        sort_by = request.args.get('sort-by', 'rating-desc')
        page = int(request.args.get('page', 1))

        min_guests = safe_int(request.form.get('guests'), 0)
        min_bedrooms = safe_int(request.form.get('bedrooms'), 0)
        min_beds = safe_int(request.form.get('beds'), 0)
        required_amenities = request.args.getlist('amenities')

    log_user_action(
        session['user_id'],
        f"Поиск: цена {min_price}-{max_price}, сортировка: {sort_by}, удобства: {required_amenities}"
    )

    all_results = filter_properties(
        min_price, max_price, min_rating, sort_by,
        min_guests=min_guests if min_guests > 0 else None,
        min_bedrooms=min_bedrooms if min_bedrooms > 0 else None,
        min_beds=min_beds if min_beds > 0 else None,
        required_amenities=required_amenities if required_amenities else None
    )

    # Пагинация
    per_page = 5
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = all_results[start:end]
    total_pages = (len(all_results) + per_page - 1) // per_page

    current_filters = {
        'min-price': min_price,
        'max-price': max_price,
        'rating': min_rating,
        'sort-by': sort_by,
        'guests': min_guests,
        'bedrooms': min_bedrooms,
        'beds': min_beds,
        'amenities': required_amenities
    }

    return render_template(
        'filter_version.html',
        results=paginated_results,
        filters=current_filters,
        current_page=page,
        total_pages=total_pages
    )

@app.route('/property/<int:property_id>')
def property_detail(property_id):

    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    log_user_action(session['user_id'], f"Просмотр объявления ID={property_id}")

    df = load_data()
    property_data = df[df['ID'] == property_id]
    if property_data.empty:
        abort(404)
    prop = property_data.iloc[0].to_dict()

    # Сохраняем оригинальные фильтры
    filters = {
        'min-price': float(request.args.get('min-price', 0)),
        'max-price': float(request.args.get('max-price', 10000)),
        'rating': float(request.args.get('rating', 0)),
        'sort-by': request.args.get('sort-by', 'rating-desc'),
        'guests': int(request.args.get('guests', 0)),
        'bedrooms': int(request.args.get('bedrooms', 0)),
        'beds': int(request.args.get('beds', 0)),
        'amenities': request.args.getlist('amenities')
    }

    amenities = prop.get('amenities', [])
    seen = set()
    cleaned = []
    for a in amenities:
        if a.startswith('Недоступно'):
            continue  # пропускаем недоступные
        clean = a.strip()
        if clean not in seen:
            seen.add(clean)
            cleaned.append(clean)
    prop['cleaned_amenities'] = cleaned

    return render_template('property_detail.html', property=prop, filters=filters)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        age = request.form.get('age')
        has_booked_before = request.form.get('has_booked_before')
        satisfaction_result = request.form.get('satisfaction_result')
        satisfaction_usability = request.form.get('satisfaction_usability')

        positive_feedback = request.form.get('positive_feedback', '').strip()
        negative_feedback = request.form.get('negative_feedback', '').strip()

        log_user_action(
            session['user_id'],
            f"Обратная связь: возраст={age}, опыт={has_booked_before}, "
            f"удовлетворённость результатом={satisfaction_result}, удобством={satisfaction_usability},"
            f"понравилось={positive_feedback}, проблемы={negative_feedback}"
        )

        return """
                <h3>Поздравляем! Вы забронировали жильё.</h3>
                <p><a href="/">Вернуться на главную</a></p>
                """

    return render_template('feedback.html')

# Модифицированный book_property, перенаправляющий на feedback
from flask import redirect, url_for

@app.route('/book/<int:property_id>', methods=['POST'])
def book_property(property_id):
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    log_user_action(
        session['user_id'],
        f"Пользователь забронировал жильё ID={property_id}"
    )

    return redirect(url_for('feedback'))


if __name__ == '__main__':
    app.run(port = 5001, debug=True)
