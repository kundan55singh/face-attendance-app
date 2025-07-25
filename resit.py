import streamlit as st
import cv2
import numpy as np
import pickle
import os

# --- CONFIG ---
DATA_DIR = "data"
HAAR_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
os.makedirs(DATA_DIR, exist_ok=True)

FACE_SIZE = (50, 50)
FACES_REQUIRED = 100
MIN_VALID_FACES = 50

# --- PAGE SETUP ---
st.set_page_config(page_title="Face Registration", layout="centered")
st.title("🧑‍💼 Face Registration System")

# --- SESSION STATE INITIALIZATION ---
if "redirect" not in st.session_state:
    st.session_state.redirect = False
if "user_registered" not in st.session_state:
    st.session_state.user_registered = ""
if "user_registered_already" not in st.session_state:
    st.session_state.user_registered_already = False

# --- REDIRECT MESSAGE ---
if st.session_state.redirect:
    if st.session_state.user_registered_already:
        st.info(f"ℹ️ '{st.session_state.user_registered}' is already registered.")
    else:
        st.success(f"✅ Registration successful for '{st.session_state.user_registered}'!")
    st.markdown("➡️ [Go to Attendance Panel](http://localhost:8502)", unsafe_allow_html=True)
    st.stop()

# --- REGISTRATION FORM ---
name = st.text_input("👤 Enter your full name")
start = st.button("📷 Start Face Registration")

# --- START CAPTURE ---
if start:
    if name.strip() == "":
        st.warning("⚠️ Please enter a valid name.")
    elif os.path.exists(os.path.join(DATA_DIR, f"{name}.pkl")):
        st.toast(f"ℹ️ User '{name}' is already registered!", icon="✅")
        st.session_state.user_registered = name
        st.session_state.user_registered_already = True
        st.session_state.redirect = True
        st.rerun()
    else:
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        faces_collected = []
        count = 0
        st.toast("📸 Capturing... Please look at the camera.")
        stframe = st.empty()

        while count < FACES_REQUIRED:
            ret, frame = cap.read()
            if not ret:
                st.error("🚫 Camera not detected.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_crop = gray[y:y + h, x:x + w]
                try:
                    resized = cv2.resize(face_crop, FACE_SIZE)
                    flattened = resized.flatten()
                    faces_collected.append(flattened)
                    count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Collected: {count}/{FACES_REQUIRED}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                except:
                    continue

            stframe.image(frame, channels="BGR")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(faces_collected) >= MIN_VALID_FACES:
            with open(os.path.join(DATA_DIR, f"{name}.pkl"), "wb") as f:
                pickle.dump(np.asarray(faces_collected), f)

            st.toast("✅ Registration successful!", icon="🎉")
            st.session_state.user_registered = name
            st.session_state.user_registered_already = False
            st.session_state.redirect = True
            st.rerun()
        else:
            st.error(f"❌ Only {len(faces_collected)} face samples captured. Try again.")
