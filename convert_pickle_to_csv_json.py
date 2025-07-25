import pickle
import pandas as pd
import json
import numpy as np
import os

# Paths
faces_pkl   = "data/faces_data.pkl"
names_pkl   = "data/names.pkl"
faces_csv   = "data/faces_data.csv"
names_json  = "data/names.json"

# ── Convert faces_data.pkl ➜ faces_data.csv ──
if os.path.exists(faces_pkl):
    with open(faces_pkl, "rb") as f:
        faces_data = pickle.load(f)

    # If it's a NumPy array
    if isinstance(faces_data, np.ndarray):
        rows = faces_data.tolist()
    # If it's a list of arrays or list-of-lists
    elif isinstance(faces_data, list):
        # convert any nested np.ndarray to list
        rows = [face.tolist() if isinstance(face, np.ndarray) else face for face in faces_data]
    else:
        rows = None

    if rows is not None:
        pd.DataFrame(rows).to_csv(faces_csv, index=False)
        print("✅ Converted faces_data.pkl ➜ faces_data.csv")
    else:
        print("❌ faces_data.pkl is not a supported format.")
else:
    print("❌ faces_data.pkl not found.")

# ── Convert names.pkl ➜ names.json ──
if os.path.exists(names_pkl):
    with open(names_pkl, "rb") as f:
        names = pickle.load(f)
    with open(names_json, "w") as f:
        json.dump(names, f)
    print("✅ Converted names.pkl ➜ names.json")
else:
    print("❌ names.pkl not found.")
