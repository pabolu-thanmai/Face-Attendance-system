from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import base64
import sqlite3
import cv2
import pickle, numpy as np
from deepface import DeepFace
import datetime

app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app)  # Enable CORS for all routes
DATABASE_FOLDER = "databases"  # Folder where databases are stored
PASSWORD = "datascience"
UPLOAD_FOLDER = "uploads"  # Folder where uploaded images are stored
CROPPED_FACES_BASE = "cropped_faces"  # Base folder for cropped faces
embeddings_path  = "Embeddings"
EMBEDDINGS_PATH = "Embeddings"
output_folder = "mean_embedding"

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create folder if not exists

def get_database_path(branch):
    """Returns the full path of the branch database."""
    return os.path.join(DATABASE_FOLDER, f"{branch}.db")

def connect_db(branch):
    """Connects to the database if it exists, else returns an error."""
    db_path = get_database_path(branch)
    if not os.path.exists(db_path):
        return None, f"Database '{branch}.db' not found!"
    return sqlite3.connect(db_path), None

def save_base64_image(image_data, img_path):
    """Decode base64 image and save it."""
    image_data = image_data.split(",")[1]  # Remove header
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(image_data))

@app.route("/upload-webcam", methods=["POST"])
def upload_webcam():
    try:
        data = request.json["image"]  # Get base64 image
        image_data = data.split(",")[1]  # Remove base64 header
        image_bytes = base64.b64decode(image_data)  # Decode base64
        
        # Save image
        image_path = os.path.join(UPLOAD_FOLDER, "webcam_capture.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        return jsonify({"message": "Webcam image saved successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload-files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in request"}), 400

    files = request.files.getlist('files')  # Get multiple files
    if not files:
        return jsonify({"error": "No files selected"}), 400

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

    return jsonify({"message": "Files uploaded successfully!"})


@app.route('/get-branches', methods=['GET'])
def get_branches():
    try:
        # Get all .db files in the database folder
        db_files = [f for f in os.listdir(DATABASE_FOLDER) if f.endswith('.db')]
        
        # Extract branch names (remove .db extension)
        branches = [db_file.replace('.db', '') for db_file in db_files]

        return jsonify({"branches": branches})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-tables', methods=['GET'])
def get_tables():
    try:
        branch = request.args.get("branch")
        if not branch:
            return jsonify({"error": "Branch not provided"}), 400

        db_path = os.path.join(DATABASE_FOLDER, f"{branch}.db")
        if not os.path.exists(db_path):
            return jsonify({"tables": []})  # No database file found

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()
        return jsonify({"tables": tables})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import re  # Import regex module

@app.route('/create-table', methods=['POST'])
def create_table():
    data = request.json
    branch = data.get("branch")
    table_name = data.get("table_name")

    if not branch or not table_name:
        return jsonify({"error": "Branch and table name are required"}), 400

    # Validate table name format: Y + 2 digits + 3 uppercase letters (e.g., Y23CSE)
    pattern = r"^Y\d{2}[A-Z]{3}$"
    if not re.match(pattern, table_name):
        return jsonify({"error": "Invalid table name format! Use 'Y23CDB' format."}), 400

    db_path = os.path.join(DATABASE_FOLDER, f"{branch}.db")
    if not os.path.exists(db_path):
        return jsonify({"error": f"Database for branch '{branch}' does not exist."}), 404

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:

        #Check for existing table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]

        # Step 4: Check if the provided table exists
        if table_name in tables:
            return jsonify({"error": f"Table '{table_name}' already exists!"})
        

        # Create table with proper constraints
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            roll_number TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)
        conn.commit()
        return jsonify({"message": f"Table '{table_name}' created successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()  
@app.route('/add-entry', methods=['POST'])
def add_entry():
    data = request.json
    branch = data.get('branch')  # This determines the database file
    table = data.get('table')  # This is the table inside that database
    roll_number = data.get('rollNumber')

    if not branch or not table or not roll_number:
        return jsonify({"error": "Branch, table, and roll number are required!"}), 400

    db_path = os.path.join(DATABASE_FOLDER, f"{branch}.db")

    if not os.path.exists(db_path):
        return jsonify({"error": f"Database for branch '{branch}' does not exist!"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if cursor.fetchone() is None:
            return jsonify({"error": f"Table '{table}' does not exist in {branch}.db! Please create it first."}), 400

        # Check if roll number already exists
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE roll_number = ?", (roll_number,))
        if cursor.fetchone()[0] > 0:
            return jsonify({"error": "Roll number already exists in this table!"}), 400

        # Insert new entry
        cursor.execute(f"INSERT INTO {table} (roll_number) VALUES (?)", (roll_number,))
        conn.commit()
        return jsonify({"message": "New entry added successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()


@app.route('/update-entry', methods=['POST'])
def update_entry():
    data = request.json
    branch = data.get('branch')  # This determines the database file
    table = data.get('table')  # This is the table inside that database
    roll_number = data.get('update_rollNumber')
    print(roll_number)
    if not branch or not table or not roll_number:
        return jsonify({"error": "Branch, table, and roll number are required!"}), 400

    db_path = os.path.join(DATABASE_FOLDER, f"{branch}.db")

    if not os.path.exists(db_path):
        return jsonify({"error": f"Database for branch '{branch}' does not exist!"}), 400

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if cursor.fetchone() is None:
            return jsonify({"error": f"Table '{table}' does not exist in {branch}.db! Please create it first."}), 400

        # Check if roll number already exists
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE roll_number = ?", (roll_number,))
        if cursor.fetchone()[0] == 0:
            return jsonify({"error": "Roll number didn't  exists in this table!"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()


@app.route("/delete-entry", methods=["POST"])
def delete_entry():
    data = request.json
    delete_type = data.get("deleteType")
    branch = data.get("branch")  # e.g., "CSD"
    table = data.get("table")  # e.g., "CSD_Y23"
    roll_number = data.get("rollNumber")  # e.g., "y22cd095"
    password = data.get("password")

    # Verify password
    if password != PASSWORD:
        return jsonify({"error": "Incorrect password!"}), 403

    conn, error = connect_db(branch)
    if error:
        return jsonify({"error": error}), 404

    cursor = conn.cursor()

    try:
        if delete_type == "database":
            conn.close()  # Close connection before deleting
            os.remove(get_database_path(branch))  # Delete the entire database file
            return jsonify({"message": f"Database '{branch}.db' deleted successfully."})

        elif delete_type == "table":
            if not table:
                return jsonify({"error": "Table name required!"}), 400
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
            return jsonify({"message": f"Table '{table}' deleted successfully."})

        elif delete_type == "roll_number":
            if not table or not roll_number:
                return jsonify({"error": "Table name and roll number required!"}), 400

            cursor.execute(f"DELETE FROM {table} WHERE roll_number = ?", (roll_number,))
            if cursor.rowcount == 0:
                return jsonify({"error": f"Roll number '{roll_number}' not found in table '{table}'."}), 404

            conn.commit()
            return jsonify({"message": f"Roll number '{roll_number}' deleted successfully from '{table}'."})

        else:
            return jsonify({"error": "Invalid delete type!"}), 400

    except sqlite3.Error as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()


def crop_faces(branch, table, roll_number):
    raw_images_path = "uploads"  # Folder with original images
    cropped_faces_path = "cropped_faces"  # Folder to save cropped faces

    # Step 1: Clear old images in cropped_faces/
    if os.path.exists(cropped_faces_path):
        for file in os.listdir(cropped_faces_path):
            file_path = os.path.join(cropped_faces_path, file)
            os.remove(file_path)
    else:
        os.makedirs(cropped_faces_path)

    # Step 2: Process each image in raw_images/
    for img_name in os.listdir(raw_images_path):
        img_path = os.path.join(raw_images_path, img_name)

        if not os.path.isfile(img_path):
            continue  # Skip non-image files

        try:
            # Detect faces using RetinaFace
            faces = DeepFace.extract_faces(img_path, detector_backend="retinaface", enforce_detection=False)

            if not faces:
                print(f"No face detected in {img_name}. Skipping...")
                continue

            # Step 3: Save each detected face
            for i, face_data in enumerate(faces):
                face = face_data["face"]
                cropped_img_path = os.path.join(cropped_faces_path, f"{img_name}_face{i}.jpg")

                # Convert face array to 0-255 range and save
                cv2.imwrite(cropped_img_path, face * 255)
                print(f"Saved cropped face: {cropped_img_path}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Face cropping complete!")
    if os.path.exists(raw_images_path):
        for file in os.listdir(raw_images_path):
            os.remove(os.path.join(raw_images_path, file))

    generate_embeddings()
    aggregate_embeddings_mean(embeddings_path, output_folder, branch, table, roll_number)


    return {"message": "Face cropping and embedding generation done!", "cropped_images_path": cropped_faces_path}

def generate_embeddings():
    cropped_faces_path = "cropped_faces"
    embeddings_path = "Embeddings"

    # Step 1: Clear old embeddings before saving new ones
    if os.path.exists(embeddings_path):
        for file in os.listdir(embeddings_path):
            os.remove(os.path.join(embeddings_path, file))
    else:
        os.makedirs(embeddings_path)

    # Step 2: Process each cropped face image
    for img_name in os.listdir(cropped_faces_path):
        img_path = os.path.join(cropped_faces_path, img_name)

        if not os.path.isfile(img_path):
            continue  # Skip non-image files

        try:
            # Generate embedding using Facenet
            embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]['embedding']
            embedding = np.array(embedding)  # Convert to NumPy array

            # Save embedding as a .pkl file
            pkl_path = os.path.join(embeddings_path, f"{img_name}.pkl")
            with open(pkl_path, "wb") as file:
                pickle.dump(embedding, file)

            print(f"Saved embedding: {pkl_path}")

        except Exception as e:
            print(f"Error generating embedding for {img_name}: {e}")

    print("Embedding generation complete!")
    if os.path.exists(cropped_faces_path):
        for file in os.listdir(cropped_faces_path):
            os.remove(os.path.join(cropped_faces_path, file))

def aggregate_embeddings_mean(embeddings_folder, output_folder, branch, tablename, rollnumber):
    """
    Computes the mean of all embeddings and saves it.
    """
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "compressed_embedding.pkl")

    embeddings = []
    for file in os.listdir(embeddings_folder):
        file_path = os.path.join(embeddings_folder, file)
        with open(file_path, "rb") as f:
            embedding = pickle.load(f)
            embeddings.append(embedding)

    if embeddings:
        mean_embedding = np.mean(embeddings, axis=0)
        with open(output_path, "wb") as f:
            pickle.dump(mean_embedding, f)
        print(f"Saved Mean embedding at {output_path}")
    else:
        print("No embeddings found!")
  
    
    # Step 2: Clear old embeddings after saving the mean embedding
    if os.path.exists(embeddings_folder):
        for file in os.listdir(embeddings_folder):
            os.remove(os.path.join(embeddings_folder, file)) 

    # Step 3: Store the compressed embedding in the database
    with open(output_path, "rb") as f:
        compressed_embedding = pickle.load(f)

    db_path = os.path.join(DATABASE_FOLDER, f"{branch}.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(f"UPDATE {tablename} SET embedding = ? WHERE roll_number = ?", 
                       (pickle.dumps(compressed_embedding), rollnumber))
        conn.commit()
        return jsonify({"message": "New entry added successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()

# Flask route to call the function
@app.route('/crop-faces', methods=['POST'])
def handle_crop_face():
    data = request.json
    branch = data.get("branch")
    table = data.get("table")
    roll_number = data.get("roll_number")

    print("\n======= API Request Received =======")
    print(f"Branch: {branch}, Table: {table}, Roll Number: {roll_number}")

    if not branch or not table or not roll_number:
        print("Error: Missing branch, table, or roll number")
        return jsonify({"error": "Missing branch, table, or roll number"}), 400

    response = crop_faces(branch, table, roll_number)
    print("API Response:", response)

    return jsonify(response)

#################################################################################################

import os
import cv2
from deepface import DeepFace

import os
import cv2
import numpy as np
import time
from deepface import DeepFace

# Function to apply histogram equalization in YCrCb color space
def histogram_equalization(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# Function to apply adaptive gamma correction
def adaptive_gamma_correction(image):
    mean_intensity = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    gamma_value = 2.0 if mean_intensity < 90 else 1.0  # Adjust dynamically
    inv_gamma = 1.0 / gamma_value
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to apply non-local means denoising
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Function to apply unsharp masking
def unsharp_masking(image):
    if image is None or image.size == 0:
        print("Error: Empty image received for sharpening.")
        return None
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    return cv2.addWeighted(image, 1.7, blurred, -0.7, 0)

# Function to apply localized sharpening (eyes & cheeks)
def localized_sharpening(image, face_rect):
    if not isinstance(face_rect, dict) or not all(k in face_rect for k in ["x", "y", "w", "h"]):
        print(f"Error: Invalid face rectangle received - {face_rect}")
        return image  # Skip processing

    fx, fy, fw, fh = face_rect["x"], face_rect["y"], face_rect["w"], face_rect["h"]

    try:
        # Ensure valid cropping
        if fx < 0 or fy < 0 or fw <= 0 or fh <= 0:
            print(f"Skipping sharpening due to invalid dimensions: {face_rect}")
            return image

        # Ensure face is a valid image
        if image is None or image.size == 0:
            print("Error: Empty image detected in localized sharpening. Skipping...")
            return None

        eye_region = image[fy:int(fy + 0.4 * fh), fx:int(fx + fw)]
        cheek_region = image[int(fy + 0.5 * fh):int(fy + 0.8 * fh), fx:int(fx + fw)]

        if eye_region is not None and eye_region.size > 0:
            eye_region = unsharp_masking(eye_region)
            image[fy:int(fy + 0.4 * fh), fx:int(fx + fw)] = eye_region

        if cheek_region is not None and cheek_region.size > 0:
            cheek_region = unsharp_masking(cheek_region)
            image[int(fy + 0.5 * fh):int(fy + 0.8 * fh), fx:int(fx + fw)] = cheek_region

    except Exception as e:
        print(f"Error during localized sharpening: {e}")

    return image


# Function to crop faces
def crop_face(image_path):
    try:
        faces = DeepFace.extract_faces(image_path, detector_backend="retinaface", enforce_detection=False)
        if not faces:
            print("No face detected")
            return [], []

        face_images = []
        face_rects = []
        for i, face in enumerate(faces):
            if "face" not in face or "facial_area" not in face:
                print(f"Error: Missing 'face' or 'facial_area' in face {i}")
                continue  # Skip this face

            face_image = (face["face"] * 255).astype(np.uint8)
            face_rect = face["facial_area"]
            
            # Debugging prints
            print(f"Face {i}: Bounding Box - {face_rect}")

            # Ensure face rectangle has valid dimensions
            if not face_rect or face_rect["w"] <= 0 or face_rect["h"] <= 0:
                print(f"Skipping face {i} due to invalid rectangle: {face_rect}")
                continue

            face_rects.append(face_rect)
            face_images.append(face_image)

        return face_images, face_rects
    except Exception as e:
        print(f"Error during face extraction: {str(e)}")
        return [], []

# Main processing function
def test_crop_faces(branch, table):
    """
    Crops faces from images in the uploads folder and applies preprocessing if intensity is low.
    """
    print("\n======= Testing Face Cropping Process Started =======")

    raw_images_path = "uploads"
    cropped_faces_path = "cropped_faces"

    # Step 1: Clear old images in cropped_faces/
    if os.path.exists(cropped_faces_path):
        for file in os.listdir(cropped_faces_path):
            file_path = os.path.join(cropped_faces_path, file)
            os.remove(file_path)
    else:
        os.makedirs(cropped_faces_path)

    # Step 2: Process each image in raw_images/
    for img_name in os.listdir(raw_images_path):
        img_path = os.path.join(raw_images_path, img_name)

        if not os.path.isfile(img_path):
            continue  # Skip non-image files

        try:
            # Detect faces using RetinaFace
            faces, face_rects = crop_face(img_path)

            if not faces:
                print(f"No face detected in {img_name}. Skipping...")
                continue

            # Step 3: Process each detected face
            for i, (face, face_rect) in enumerate(zip(faces, face_rects)):
                cropped_img_path = os.path.join(cropped_faces_path, f"{img_name}_face{i}.jpg")
                # Convert face array to 0-255 range and calculate intensity
                mean_intensity = np.mean(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))

                if mean_intensity < 85:
                    print(f"Face {i+1} is too dark (Intensity: {mean_intensity:.2f}). Applying enhancements...")
                    face = histogram_equalization(face)
                    face = adaptive_gamma_correction(face)
                    face = denoise_image(face)
                    face = unsharp_masking(face)
                    face = localized_sharpening(face, face_rect)

                    cropped_img_path = os.path.join(cropped_faces_path, f"{img_name}_face{i}_dark.jpg")

                    cv2.imwrite(cropped_img_path, face)
                    print(f"Saved cropped face: {cropped_img_path}")
                else:
                    print(f"Face {i+1} is bright enough (Intensity: {mean_intensity:.2f}). No enhancement applied.")
                    cropped_img_path = os.path.join(cropped_faces_path, f"{img_name}_face{i}.jpg")
                    cv2.imwrite(cropped_img_path, face)
                    print(f"Saved cropped face: {cropped_img_path}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Face cropping complete!")

    print("\n======= Face Cropping Process Completed =======")
    if os.path.exists(raw_images_path):
        for file in os.listdir(raw_images_path):
            os.remove(os.path.join(raw_images_path, file))

    generate_and_match_embeddings("cropped_faces", branch, table)



    return {"message": "Face cropping and embedding generation done!", "cropped_images_path": cropped_faces_path}

from scipy.spatial.distance import cosine
from deepface import DeepFace

def generate_and_match_embeddings(cropped_faces_folder, branch, table, similarity_threshold=0.70):
    """
    Generates embeddings for cropped faces, searches them in the specified branch.db and table,
    and returns roll numbers with cosine similarity > threshold.


    Returns:
    - List of matched roll numbers.
    """
    print("\n======= Generating and Matching Embeddings =======")

    # Step 1: Check and prepare database path
    db_path = os.path.join(DATABASE_FOLDER, f"{branch}.db")
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found! Exiting...")
        return []

    matched_roll_numbers = []

    # Step 2: Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Step 3: Retrieve stored embeddings from the database
    cursor.execute(f"SELECT roll_number, embedding FROM {table} WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()

    if not rows:
        print("No stored embeddings found in database!")
        conn.close()
        return []
    rows_dict = {roll_number: pickle.loads(embedding) for roll_number, embedding in rows}

    # Step 4: Process each cropped face image
  


    for img_name in os.listdir(cropped_faces_folder):
        img_path = os.path.join(cropped_faces_folder, img_name)
        similarity_threshold = 0.70

        if not os.path.isfile(img_path):
            continue  # Skip non-image files
        if "dark" in img_path:
            similarity_threshold = 0.65

        try:
            # Generate embedding using FaceNet
            embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]['embedding']

            print(f"\nProcessing: {img_name}")
            print(f"Generated embedding for {img_name}")

            max_similarity = 0  # Track the highest similarity
            best_match = None  # Track the best roll number

            # Compare with stored embeddings
            for roll_number, stored_embedding in rows_dict.items():
                
                try:
                    # Deserialize stored embedding using pickle.loads()
                    #stored_embedding = pickle.loads(stored_embedding) #Additional statement
                    # Compute cosine similarity

                    similarity = 1 - cosine(embedding, stored_embedding)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = roll_number  # Update the best match

                except Exception as e:
                    print(f"Error comparing embeddings for Roll {roll_number}: {e}")

            # Append only the best match for this image if it meets the threshold
            if best_match and max_similarity > similarity_threshold:
                print(f"Best Matched Roll Number for {img_name}: {best_match} (Similarity: {max_similarity:.2f})")
                matched_roll_numbers.append(best_match)
                # if similarity_threshold >= 0.60:
                del rows_dict[best_match]
                

        except Exception as e:
            print(f"Error processing {img_name}: {e}")


    # Step 6: Delete processed cropped faces
   # for file in os.listdir(cropped_faces_folder):
    #    os.remove(os.path.join(cropped_faces_folder, file))

    print("\n======= Matching Completed =======")
    conn.close()
    ATTENDANCE_FOLDER = "Attendance"
    matched_roll_numbers = list(set(matched_roll_numbers))
    print("Matched Roll Numbers:", matched_roll_numbers)

    # Step 6: Update Attendance Database
    attendance_db_path = os.path.join(ATTENDANCE_FOLDER, f"{branch}Attendance.db")
    
    if not os.path.exists(attendance_db_path):
        print(f"Attendance database {attendance_db_path} not found! Exiting...")
        return

    conn = sqlite3.connect(attendance_db_path)
    cursor = conn.cursor()

    attendance_table = f"{table}"
    print(table)

    # Mark matched roll numbers as "Present"
    for roll_number in matched_roll_numbers:
        cursor.execute(f"UPDATE {attendance_table} SET attendance='Present' WHERE roll_number=?", (roll_number,))
    
    conn.commit()

    # Step 7: Fetch and Print Absent Students
    today_date = datetime.date.today().strftime("%Y-%m-%d")

    print("\n======= Absent Students =======")

    print(f"{today_date}")
    print("-" * 10)

    cursor.execute(f"SELECT roll_number FROM {attendance_table} WHERE attendance='Absent'")
    absent_students = cursor.fetchall()

    for student in absent_students:
        print(student[0])
    
    for roll_number in matched_roll_numbers:
        cursor.execute(f"UPDATE {attendance_table} SET attendance='Absent' WHERE roll_number=?", (roll_number,))
    conn.commit()
    conn.close()

# Flask route to call the function
@app.route('/check-faces', methods=['POST'])
def handle_crop_faces():
    data = request.json
    branch = data.get("branch")
    table = data.get("table")
    print(branch, table)

    print("\n======= API Request Received =======")
    print(f"Branch: {branch}, Table: {table}")

    if not branch or not table :
        print("Error: Missing branch, table, or roll number")
        return jsonify({"error": "Missing branch, table, or roll number"}), 400

    response = test_crop_faces(branch, table)
    print("API Response:", response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)