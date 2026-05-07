import streamlit as st
import os
import sqlite3
from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# ==============================
# Streamlit Grundeinstellung
# ==============================

st.set_page_config(
    page_title="Schul-Fundbüro",
    layout="wide"
)

# ==============================
# Modernes Styling
# ==============================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stApp {
    background-color: #121212;
}

.fund-card {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    text-align: center;
}

.fund-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 10px;
    color: white;
}

.fund-date {
    color: #AAAAAA;
    font-size: 14px;
}

.fund-confidence {
    color: #4CAF50;
    font-weight: bold;
    margin-top: 5px;
}

img {
    border-radius: 12px;
}

.stButton > button {
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Konfiguration
# ==============================

IMAGE_FOLDER = "images"
DB_PATH = "fundbuero.db"
ADMIN_PASSWORD = "admin123"

# ==============================
# Initialisierung
# ==============================

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            category TEXT,
            confidence REAL,
            upload_date TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ==============================
# YOLOv8 Modell laden
# ==============================

@st.cache_resource
def load_ai_model():
    try:
        model = YOLO("yolov8n.pt")
        return model

    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None

model = load_ai_model()

# ==============================
# KI-Vorhersage
# ==============================

def predict_image(image):

    results = model(image)
    result = results[0]

    if len(result.boxes) == 0:
        return "Unbekannt", 0.0

    # Beste Box mit höchster Konfidenz wählen
    best_box = max(
        result.boxes,
        key=lambda b: float(b.conf[0])
    )

    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])

    class_name = model.names[class_id]

    return class_name, confidence

# ==============================
# Datenbankfunktionen
# ==============================

def insert_item(filename, category, confidence):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT INTO items (
            filename,
            category,
            confidence,
            upload_date
        )
        VALUES (?, ?, ?, ?)
    """, (
        filename,
        category,
        confidence,
        datetime.now().strftime("%d.%m.%Y %H:%M")
    ))

    conn.commit()
    conn.close()

def get_items_by_category(category):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT * FROM items
        WHERE category=?
        ORDER BY id DESC
    """, (category,))

    items = c.fetchall()

    conn.close()

    return items

def get_all_items():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT * FROM items
        ORDER BY id DESC
    """)

    items = c.fetchall()

    conn.close()

    return items

def delete_item(item_id, filename):

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        "DELETE FROM items WHERE id=?",
        (item_id,)
    )

    conn.commit()
    conn.close()

    image_path = os.path.join(
        IMAGE_FOLDER,
        filename
    )

    if os.path.exists(image_path):
        os.remove(image_path)

def get_total_count():

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM items")

    count = c.fetchone()[0]

    conn.close()

    return count

# ==============================
# UI
# ==============================

st.title("🎒 Digitales Schul-Fundbüro")

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "Finder (Upload)",
        "Verloren & Suchen",
        "Admin"
    ]
)

# ==============================
# 1. Upload
# ==============================

if menu == "Finder (Upload)":

    st.header("📤 Gegenstand hochladen")

    st.info(
        "Bitte nur Gegenstände fotografieren "
        "und keine Personen."
    )

    uploaded_file = st.file_uploader(
        "Bild hochladen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None and model is not None:

        try:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

            # Bild verkleinern
            image.thumbnail((800, 800))

            st.image(
                image,
                caption="Hochgeladenes Bild",
                width=300
            )

            if st.button(
                "🔍 Klassifizieren und speichern"
            ):

                with st.spinner(
                    "KI analysiert das Bild..."
                ):

                    category, confidence = predict_image(image)

                confidence_percent = round(
                    confidence * 100,
                    2
                )

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                filename = f"{timestamp}.jpg"

                image.save(
                    os.path.join(
                        IMAGE_FOLDER,
                        filename
                    )
                )

                insert_item(
                    filename,
                    category,
                    confidence_percent
                )

                st.success(
                    "✅ Gegenstand gespeichert!"
                )

                st.markdown(
                    f"### 📦 {category}"
                )

                st.progress(
                    confidence_percent / 100
                )

                st.caption(
                    f"KI-Sicherheit: "
                    f"{confidence_percent}%"
                )

        except Exception:
            st.error(
                "Fehler beim Laden des Bildes."
            )

# ==============================
# 2. Suche
# ==============================

elif menu == "Verloren & Suchen":

    st.header("🔍 Nach Gegenständen suchen")

    st.write(
        f"Gesamtzahl gespeicherter Gegenstände: "
        f"**{get_total_count()}**"
    )

    categories = (
        list(model.names.values())
        if model else []
    )

    selected_category = st.selectbox(
        "Kategorie auswählen",
        categories
    )

    if selected_category:

        items = get_items_by_category(
            selected_category
        )

        if items:

            # 3 Karten pro Zeile
            cols = st.columns(3)

            for index, item in enumerate(items):

                image_path = os.path.join(
                    IMAGE_FOLDER,
                    item[1]
                )

                with cols[index % 3]:

                    st.markdown(
                        '<div class="fund-card">',
                        unsafe_allow_html=True
                    )

                    st.image(
                        image_path,
                        use_container_width=True
                    )

                    st.markdown(
                        f'''
                        <div class="fund-title">
                        📦 {item[2]}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'''
                        <div class="fund-confidence">
                        KI: {item[3]}%
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'''
                        <div class="fund-date">
                        {item[4]}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

        else:
            st.info(
                "🔍 Keine Gegenstände gefunden."
            )

# ==============================
# 3. Admin
# ==============================

elif menu == "Admin":

    st.header("⚙️ Admin-Bereich")

    password = st.text_input(
        "Passwort",
        type="password"
    )

    if password == ADMIN_PASSWORD:

        st.success("Zugriff gewährt")

        items = get_all_items()

        for item in items:

            col1, col2 = st.columns([3, 1])

            with col1:

                st.image(
                    os.path.join(
                        IMAGE_FOLDER,
                        item[1]
                    ),
                    caption=(
                        f"{item[2]} | "
                        f"{item[3]}% | "
                        f"{item[4]}"
                    ),
                    width=250
                )

            with col2:

                if st.button(
                    f"🗑️ Löschen ID {item[0]}"
                ):

                    delete_item(
                        item[0],
                        item[1]
                    )

                    st.warning(
                        "Eintrag gelöscht."
                    )

                    st.rerun()

    elif password != "":

        st.error("Falsches Passwort.")
