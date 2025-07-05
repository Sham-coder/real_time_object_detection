import os
import cv2
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, Response, g, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

DATABASE = 'users.db'

# Database helper functions
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
        db.commit()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Routes for signup, login, logout
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash('Please fill out both fields')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password)
        try:
            db = get_db()
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            db.commit()
            flash('User created successfully! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
            return redirect(url_for('signup'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('detection'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['new_password']
        if not username or not new_password:
            flash('Please fill out both fields')
            return redirect(url_for('forgot_password'))
        user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
        if user:
            hashed_password = generate_password_hash(new_password)
            db = get_db()
            db.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_password, username))
            db.commit()
            flash('Password updated successfully! Please login.')
            return redirect(url_for('login'))
        else:
            flash('Username not found')
            return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

# Object detection setup
thres = 0.5
nms_threshold = 0.2

classNames = []
with open('objects.txt','r') as f:
    classNames = f.read().splitlines()

Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

cap = None

latest_detected_objects = []

def gen_frames():
    global cap, latest_detected_objects
    if cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
        cap.set(cv2.CAP_PROP_BRIGHTNESS,100)
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            classIds, confs, bbox = net.detect(img,confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1,-1)[0])
            confs = list(map(float,confs))
            indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
            detected_objects = []
            if len(classIds) != 0:
                for i in indices:
                    i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                    label = classNames[classIds[i]-1]
                    if label.lower() == "dog":
                        continue
                    box = bbox[i]
                    color = Colors[classIds[i]-1]
                    confidence = str(round(confs[i],2))
                    x,y,w,h = box[0],box[1],box[2],box[3]
                    cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
                    cv2.putText(img, label+" "+confidence,(x+10,y+20),
                                cv2.FONT_HERSHEY_PLAIN,1,color,2)
                    detected_objects.append(f"{label} ({confidence})")
            latest_detected_objects = detected_objects
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            # Yield frame in byte format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detected_objects')
def detected_objects():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return {"objects": latest_detected_objects}

@app.route('/detection')
def detection():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('detection.html', username=session.get('username'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
