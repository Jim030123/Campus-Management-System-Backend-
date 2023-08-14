from flask import Flask, render_template, Response, jsonify
import os
import util

import cv2
import numpy as np
import easyocr
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import firestore

from datetime import datetime

from firebase_admin import storage


cred = credentials.Certificate(r'yolov3-from-opencv-object-detection\firebase_credentials.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'campus-management-system-4962b.appspot.com'})
db = firestore.client()




app = Flask(__name__)

logbook_directory = "logbook"
os.makedirs(logbook_directory, exist_ok=True)

# 定义常量
model_cfg_path = os.path.join('.', 'yolov3-from-opencv-object-detection\model\cfg\darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model_weight\model.weights')
class_names_path = os.path.join('.', 'yolov3-from-opencv-object-detection\model\class.names')
logbook_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'  # Generate a unique logbook filename based on the current date and time
logbook_path = os.path.join(logbook_directory, logbook_filename)  

# 加载类别名称
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
    f.close()

# 加载模型
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# 初始化OCR阅读器
reader = easyocr.Reader(['en'])

# 初始化车牌号码列表
car_plate_numbers = []

# 视频流函数
def generate_frames():
    cap = cv2.VideoCapture(0)  # 0代表默认摄像头
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape

        # 转换图像（精度）
        blob = cv2.dnn.blobFromImage(frame, 1 / 1000, (320, 320), (0, 0, 0), True)

        # 获取检测结果
        net.setInput(blob)
        detections = util.get_outputs(net)

        # 边界框、类别ID和置信度
        bboxes = []
        class_ids = []
        scores = []

        for detection in detections:
            # [x1, x2, x3, x4, x5, x6, ..., x85]
            bbox = detection[:4]

            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

            bbox_confidence = detection[4]

            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])

            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        # 应用NMS
        bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

        # 处理车牌号码
        for bbox_, bbox in enumerate(bboxes):
            xc, yc, w, h = bbox

            cv2.putText(frame,
                        class_names[class_ids[bbox_]],
                        (int(xc - (w / 2)), int(yc + (h / 2) - 100)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        7)

            license_plate = frame[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

            frame = cv2.rectangle(frame,
                                  (int(xc - (w+1 / 2)), int(yc - (h+1 / 2))),
                                  (int(xc + (w+1 / 2)), int(yc + (h+1 / 2))),
                                  (0, 255, 0),
                                  2)
            # 使用灰度版本
            license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

            _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            output = reader.readtext(license_plate_gray)

            for out in output:
                text_bbox, text, text_score = out
                if text_score > 0.4:
                    car_plate_number = text.replace(' ', '')
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current timestamp
                    result_in_firebase = check_data_in_firebase(car_plate_number)
                    car_plate = {
                        'number': car_plate_number,
                        'timestamp': timestamp,
                        'result_in_firebase': result_in_firebase
                    }

                    car_plate_numbers.append(car_plate)

                    # Save the car plate number and timestamp to the logbook file
                    with open(logbook_path, 'a') as logbook_file:
                        logbook_file.write(f"{timestamp} - Car Plate Number: {car_plate_number} \nGate Status (if True gate open): {result_in_firebase}\n")

                    cv2.putText(frame,
                                car_plate_number,
                                (int(xc - (w / 2)), int(yc + (h / 2)-150)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                (0, 0, 255),
                                7)
                    print(car_plate_number)  # 打印车牌号码到控制台
                    # if car_plate_number not in car_plate_numbers:
                    # car_plate_numbers.append(car_plate_number)  # 将车牌号码添加到列表中

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 拼接帧并显示结果

    cap.release()

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/car_plate_number')
def get_car_plate_number():
    # Read the logbook file content
    with open(logbook_path, 'r') as logbook_file:
        logbook_content = logbook_file.read()

    return jsonify(car_plate_numbers=car_plate_numbers, logbook_content=logbook_content)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect1')
def detect1():
    latest_car_plate = car_plate_numbers[-1] if car_plate_numbers else {}
    latest_car_plate_number = latest_car_plate.get('number', "")
    latest_car_plate_timestamp = latest_car_plate.get('timestamp', "")
    is_data_in_firebase = check_data_in_firebase(latest_car_plate_number)
    image_url = get_image_url_from_firebase(latest_car_plate_number)
    return render_template('detect1.html', car_plate_number=latest_car_plate_number, 
                           timestamp=latest_car_plate_timestamp, is_data_in_firebase=is_data_in_firebase,
                           image_url=image_url)



def upload_logbook_to_firebase():
    bucket = storage.bucket()

    try:
        # Upload the logbook file
        blob = bucket.blob("logbooks/" + logbook_filename)
        blob.upload_from_filename(logbook_path)
        print(f"Logbook uploaded to Firebase Storage as '{logbook_filename}'.")
    except Exception as e:
        print(f"Error uploading logbook to Firebase Storage: {e}")


def check_data_in_firebase(data):
    doc_ref = db.collection('vehicle').document(data)
    doc = doc_ref.get()
    if doc.exists:
        # Check if the "status" field exists in the document
        if "status" in doc.to_dict():
            # Check if the "status" field is equal to "approved"
            if doc.to_dict()["status"] == "Approved":
                return True
    return False


def get_image_url_from_firebase(data):
    doc_ref = db.collection('vehicle').document(data)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("photoUrl", "")
    return ""

@app.route('/logbooks')
def list_logbooks():
    bucket = storage.bucket()
    logbooks_folder = "logbooks/"
    blobs = bucket.list_blobs(prefix=logbooks_folder)

    # Extract the list of text files in the logbooks folder
    text_files = [blob.name.split("/")[-1] for blob in blobs if blob.name.endswith(".txt")]

    return render_template('logbooks.html', text_files=text_files)



@app.route('/logbooks/<string:filename>')
def view_logbook(filename):
    bucket = storage.bucket()
    logbooks_folder = "logbooks/"
    blob = bucket.blob(logbooks_folder + filename)

    try:
        # Download the content of the selected logbook file from Firebase Storage
        logbook_content = blob.download_as_text()
    except Exception as e:
        logbook_content = f"Error retrieving logbook content: {e}"

    return render_template('logbook_content.html', filename=filename, logbook_content=logbook_content)



    
if __name__ == '__main__':
    # Create the logbook file
    


  
    app.run(debug=True)  
    upload_logbook_to_firebase()
    