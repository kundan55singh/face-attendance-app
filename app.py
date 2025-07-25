import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import pickle
import json
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# ---------- CONFIGURATION ----------
DATA_DIR = "data"
ATTENDANCE_DIR = "attendance"
CSV_PATH = os.path.join(ATTENDANCE_DIR, "attendance.csv")
FACE_MODEL_PATH = os.path.join(DATA_DIR, "faces_data.pkl")
NAME_MODEL_PATH = os.path.join(DATA_DIR, "names.pkl")
HAAR_MODEL_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")
USER_DB_PATH = os.path.join(DATA_DIR, "users.csv")

FACE_SIZE = (75, 100)
N_SAMPLES = 20 # Number of face samples to capture

faces_csv_path = os.path.join(DATA_DIR, "faces_data.csv")
names_json_path = os.path.join(DATA_DIR, "names.json")

if os.path.exists(FACE_MODEL_PATH):
    with open(FACE_MODEL_PATH, "rb") as f:
        faces_data_check = pickle.load(f)
    if isinstance(faces_data_check, np.ndarray):
        pd.DataFrame(faces_data_check.tolist()).to_csv(faces_csv_path, index=False)
        print("✅ Converted faces_data.pkl to faces_data.csv")
    elif isinstance(faces_data_check, list):
        pd.DataFrame(faces_data_check).to_csv(faces_csv_path, index=False)
        print("✅ Converted faces_data.pkl to faces_data.csv (list)")
    else:
        print("❌ faces_data.pkl is not a supported format.")

if os.path.exists(NAME_MODEL_PATH):
    with open(NAME_MODEL_PATH, "rb") as f:
        names_check = pickle.load(f)
    with open(names_json_path, "w") as f:
        json.dump(names_check, f)
    print("✅ Converted names.pkl to names.json")

# ---------- INITIALIZATION ----------
st.set_page_config(page_title="Face Attendance System", layout="wide")

# Create necessary directories
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Create files if they don't exist
if not os.path.exists(CSV_PATH):
    pd.DataFrame(columns=["Name", "Date", "Time", "Status"]).to_csv(CSV_PATH, index=False)
if not os.path.exists(USER_DB_PATH):
    pd.DataFrame([["admin", "admin"]], columns=["Name", "Role"]).to_csv(USER_DB_PATH, index=False)

# Initialize session state variables
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "role" not in st.session_state:
    st.session_state.role = None
if "new_user_name" not in st.session_state:
    st.session_state.new_user_name = ""
if "new_user_role" not in st.session_state:
    st.session_state.new_user_role = "user"
if "captured_faces" not in st.session_state:
    st.session_state.captured_faces = []


# ---------- LOAD MODELS AND DATA ----------
try:
    facedetect = cv2.CascadeClassifier(HAAR_MODEL_PATH)
    if os.path.exists(FACE_MODEL_PATH) and os.path.exists(NAME_MODEL_PATH):
        with open(FACE_MODEL_PATH, 'rb') as f:
            faces_data = pickle.load(f)
        with open(NAME_MODEL_PATH, 'rb') as f:
            names = pickle.load(f)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        knn.fit(np.array(faces_data), np.array(names))
    else:
        faces_data, names = [], []
        knn = None
        st.sidebar.warning("⚠️ Face model not found. Please register users.")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()


# ---------- CORE FUNCTIONS ----------
def log_attendance(name, status):
    """Logs the user's attendance, checking for duplicates for the same day and status."""
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")
    df = pd.read_csv(CSV_PATH)
    already_logged = df[(df["Name"] == name) & (df["Date"] == date_str) & (df["Status"] == status)]
    if already_logged.empty:
        new_row = pd.DataFrame([[name, date_str, time_str, status]], columns=["Name", "Date", "Time", "Status"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        st.success(f"✅ Attendance Marked: **{name}** as **{status}** at {time_str}.")
    else:
        st.info(f"ℹ️ **{name}** has already been marked as **{status}** today.")

def delete_user_and_face_data(user_df_to_save, user_to_delete):
    """Saves the updated user CSV and removes the specified user's face data from pickle files."""
    user_df_to_save.to_csv(USER_DB_PATH, index=False)
    try:
        if not os.path.exists(FACE_MODEL_PATH): return

        with open(FACE_MODEL_PATH, 'rb') as f: current_faces = list(pickle.load(f))
        with open(NAME_MODEL_PATH, 'rb') as f: current_names = list(pickle.load(f))
        
        new_faces, new_names = [], []
        for face, name in zip(current_faces, current_names):
            if name != user_to_delete:
                new_faces.append(face)
                new_names.append(name)
                
        with open(FACE_MODEL_PATH, 'wb') as f: pickle.dump(new_faces, f)
        with open(NAME_MODEL_PATH, 'wb') as f: pickle.dump(new_names, f)
    except Exception as e:
        st.error(f"Error updating face data files: {e}")

def update_user_data(user_df_to_save, old_name, new_name):
    """Saves the user CSV and updates the user's name in the face data pickle file."""
    user_df_to_save.to_csv(USER_DB_PATH, index=False)
    try:
        if not os.path.exists(NAME_MODEL_PATH) or old_name == new_name: return

        with open(NAME_MODEL_PATH, 'rb') as f: current_names = list(pickle.load(f))
        
        updated_names = [new_name if name == old_name else name for name in current_names]
        
        with open(NAME_MODEL_PATH, 'wb') as f: pickle.dump(updated_names, f)
    except Exception as e:
        st.error(f"Error updating name data file: {e}")

# ---------- STREAMLIT UI ----------
st.title("🧑‍💼 Face Recognition Attendance System")
st.markdown("---")

tabs = st.tabs(["📸 Mark Attendance", "🛡️ Admin Panel"])

# ---------- TAB 1: MARK ATTENDANCE ----------
with tabs[0]:
    st.header("📍 Mark Your Attendance")
    col1, col2 = st.columns([2, 3])
    with col1:
        status = st.radio("Select Action:", ["Punch In", "Punch Out"], horizontal=True)
        img_file_buffer = st.camera_input("Click your picture to mark attendance", key="attendance_cam")
    
    if img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        img_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray_img, 1.3, 5)

        if len(faces) == 0:
            st.warning("⚠️ No face detected. Please try again.")
        elif knn is None:
            st.error("❌ Face recognition model is not trained. Admin must register users.")
        else:
            for (x, y, w, h) in faces:
                face_crop = gray_img[y:y+h, x:x+w]
                resized_face = cv2.resize(face_crop, FACE_SIZE).flatten().reshape(1, -1)
                try:
                    predicted_name = knn.predict(resized_face)[0]
                    log_attendance(predicted_name, status)
                    cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img_array, str(predicted_name), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                except Exception as e:
                    st.error(f"An error occurred during face prediction: {e}")
            
            with col2:
                st.image(img_array, channels="BGR", caption="Verified Image")

# ---------- TAB 2: ADMIN PANEL ----------
with tabs[1]:
    st.header("🔐 Admin Panel")
    if st.session_state.logged_in_user is None:
        with st.container(border=True):
            st.subheader("Admin Login")
            admin_name_input = st.text_input("Admin Name", key="admin_login_name")
            if st.button("Login"):
                user_df = pd.read_csv(USER_DB_PATH)
                admin_user = user_df[(user_df["Name"] == admin_name_input) & (user_df["Role"].str.lower() == "admin")]
                if not admin_user.empty:
                    st.session_state.logged_in_user = admin_name_input
                    st.session_state.role = "admin"
                    st.success(f"✅ Welcome, Admin {admin_name_input}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid admin name or you do not have admin privileges.")
    else:
        st.success(f"Logged in as Admin: **{st.session_state.logged_in_user}**")
        if st.button("Logout"):
            st.session_state.logged_in_user = None
            st.session_state.role = None
            st.session_state.captured_faces = []
            st.session_state.new_user_name = ""
            st.rerun()
        st.markdown("---")
        
        # --- REGISTER NEW USER ---
        with st.expander("➕ Register New User", expanded=True):
            user_df = pd.read_csv(USER_DB_PATH)
            
            st.session_state.new_user_name = st.text_input("Enter New User's Name:", value=st.session_state.new_user_name, key="new_user_name_input")
            st.session_state.new_user_role = st.selectbox("Assign Role:", ["user", "admin"], index=0 if st.session_state.new_user_role=="user" else 1, key="new_user_role_select")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Samples Captured: {len(st.session_state.captured_faces)} / {N_SAMPLES}**")
                progress_value = min(1.0, len(st.session_state.captured_faces) / N_SAMPLES)
                st.progress(progress_value)
            with col2:
                if st.button("🔄 Clear/Restart"):
                    st.session_state.captured_faces = []
                    st.session_state.new_user_name = ""
                    st.rerun()
            
            # Logic to prevent capturing more photos than needed
            if len(st.session_state.captured_faces) < N_SAMPLES:
                img_capture = st.camera_input("Take a photo for the user's profile", key="register_cam")
                if img_capture:
                    if not st.session_state.new_user_name:
                        st.error("Please enter a name for the user before taking photos.")
                    elif st.session_state.new_user_name in user_df["Name"].values:
                        st.error("A user with this name already exists.")
                    else:
                        bytes_data = img_capture.getvalue()
                        img_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                        gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                        faces_detected = facedetect.detectMultiScale(gray_img, 1.3, 5)
                        
                        if len(faces_detected) != 1:
                            st.warning("⚠️ Please ensure exactly one face is in the photo.")
                        else:
                            (x, y, w, h) = faces_detected[0]
                            face_crop = gray_img[y:y+h, x:x+w]
                            resized_face = cv2.resize(face_crop, FACE_SIZE).flatten()
                            st.session_state.captured_faces.append(resized_face)
                            st.rerun()
            else:
                st.success(f"✅ All {N_SAMPLES} samples have been captured for **{st.session_state.new_user_name}**!")
                if st.button("💾 Save User and Face Data"):
                    new_faces = st.session_state.captured_faces
                    new_name = st.session_state.new_user_name
                    
                    faces_data.extend(new_faces)
                    names.extend([new_name] * len(new_faces))
                    
                    with open(FACE_MODEL_PATH, 'wb') as f: pickle.dump(faces_data, f)
                    with open(NAME_MODEL_PATH, 'wb') as f: pickle.dump(names, f)
                    
                    new_user_row = pd.DataFrame([[new_name, st.session_state.new_user_role]], columns=["Name", "Role"])
                    user_df = pd.concat([user_df, new_user_row], ignore_index=True)
                    user_df.to_csv(USER_DB_PATH, index=False)
                    
                    st.success(f"User '{new_name}' registered successfully!")
                    
                    # Reset state
                    st.session_state.captured_faces = []
                    st.session_state.new_user_name = ""
                    st.rerun()

        # --- VIEW ATTENDANCE & MANAGE USERS ---
        with st.expander("📅 View Attendance Records"):
            try:
                st.dataframe(pd.read_csv(CSV_PATH), use_container_width=True)
            except FileNotFoundError:
                st.info("No attendance records yet.")
        
        with st.expander("👥 Manage Registered Users"):
            user_df_manage = pd.read_csv(USER_DB_PATH)
            if user_df_manage.empty:
                st.info("No users registered.")
            else:
                for i, row in user_df_manage.iterrows():
                    # Use setdefault to initialize editing state for each user
                    st.session_state.setdefault(f'editing_{i}', False)

                    if st.session_state[f'editing_{i}']:
                        # --- EDITING VIEW ---
                        with st.container(border=True):
                            c1, c2 = st.columns(2)
                            new_name = c1.text_input("Name", value=row['Name'], key=f"edit_name_{i}")
                            new_role = c2.selectbox("Role", ["user", "admin"], index=0 if row['Role']=='user' else 1, key=f"edit_role_{i}")
                            
                            c3, c4, c5 = st.columns([1, 1, 5])
                            if c3.button("💾 Save", key=f"save_{i}"):
                                old_name = row['Name']
                                user_df_manage.at[i, 'Name'] = new_name
                                user_df_manage.at[i, 'Role'] = new_role
                                
                                # Check if we are demoting the last admin
                                admins = user_df_manage[user_df_manage['Role'].str.lower() == 'admin']
                                if old_name.lower() == 'admin' and len(admins) == 0:
                                     st.error("Cannot remove the last admin role.")
                                else:
                                    update_user_data(user_df_manage, old_name=old_name, new_name=new_name)
                                    st.session_state[f'editing_{i}'] = False
                                    st.success(f"User '{old_name}' updated successfully!")
                                    st.rerun()

                            if c4.button("❌ Cancel", key=f"cancel_{i}"):
                                st.session_state[f'editing_{i}'] = False
                                st.rerun()
                    else:
                        # --- DEFAULT VIEW ---
                        c1, c2, c3 = st.columns([4, 1, 1])
                        role_icon = "👑" if row['Role'].lower() == 'admin' else '👤'
                        c1.write(f"{role_icon} **{row['Name']}** (Role: {row['Role']})")
                        
                        if c2.button("✏️ Edit", key=f"edit_{i}"):
                            st.session_state[f'editing_{i}'] = True
                            st.rerun()
                            
                        if c3.button("🗑️ Delete", key=f"delete_{i}"):
                            admins = user_df_manage[user_df_manage['Role'].str.lower() == 'admin']
                            if row['Role'].lower() == 'admin' and len(admins) <= 1:
                                st.error("❌ Cannot delete the last admin.")
                            else:
                                user_to_delete = row['Name']
                                user_df_manage = user_df_manage.drop(i)
                                delete_user_and_face_data(user_df_manage, user_to_delete)
                                st.success(f"✅ User '{user_to_delete}' has been deleted.")
                                st.rerun()