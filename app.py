# Standard library imports
from datetime import datetime
import csv
import os
import shutil
import subprocess
import time
import difflib

# Third-party imports
from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, flash, Response, session, stream_with_context)
import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import pytesseract
from easyocr import Reader
from paddleocr import PaddleOCR
import numpy as np

# Global variables and configurations
app = Flask(__name__)
app.config['SECRET_KEY'] = "my_flask_application"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'out'
app.config['DET_CONF'] = 0.5
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'out'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'zip', 'mp4'}
valid_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- '
global_frames = None
ocr_conf_global = 0.5
det_conf_global = 0.5
live_data = {}
current_ocr_choice = None
csv_filename = f'record/records-{time.time()}.csv'
total_detected = 0
total_detected_wo_rtsp = 0
count_in_parking = 0
count_in_roadway = 0
successful_ocr = 0
unsuccessful_ocr = 0
successful_ocr_ids = set()
processed_ids = set()
counted_vehicle_ids = set()

# Initialize the YOLOv8 model at the global level

model_vehicle = YOLO('weights/yolov8x.pt')
model = YOLO('weights/best.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


reader = Reader(['en'], gpu=True) 

# Create the PaddleOCR instance
paddle_ocr = PaddleOCR(
    warmup=True,           # Preload the model to reduce initial inference time.
    use_tensorrt=True,     # Use TensorRT for optimization if you're using an NVIDIA GPU.
    use_gpu=True,          # Use GPU for inference if available.
    det_limit_side_len=640, # Reduce for faster processing at the cost of some accuracy.
    det_algorithm='DB',    # Use the DB algorithm for detection, which is fast.
    enable_mkldnn=True,    # Use MKL-DNN optimization for CPU inference.
    use_mp=True,           # Use multi-process for parallel inference if batching images.
    use_angle_cls=False    # Set to False if your text is always upright.
)

def clear_global_variables():
    global total_detected, successful_ocr, unsuccessful_ocr, count_in_parking, count_in_roadway, total_detected_wo_rtsp
    total_detected = 0
    successful_ocr = 0
    unsuccessful_ocr = 0
    count_in_parking = 0
    count_in_roadway = 0
    total_detected_wo_rtsp = 0
    
parking_1_coords = [
    (3.967532467532436, 237.00649350649343),
    (438.77272727272725, 73.37012987012974),
    (360.8506493506493, 42.20129870129858),
    (5.525974025973994, 135.70779220779207),
    (5.525974025973994, 237.00649350649343)
]

road_way_coords = [
    (8.64285714285711, 237.00649350649343),
    (490.2012987012987, 62.46103896103887),
    (536.9545454545454, 84.27922077922074),
    (519.8116883116883, 99.86363636363626),
    (502.66883116883116, 113.88961038961031),
    (524.4870129870129, 132.590909090909),
    (10.201298701298668, 453.62987012987),
    (8.64285714285711, 244.7987012987012)
]

parking_2_coords = [
    (499.551948051948, 109.21428571428567),
    (533.8376623376623, 137.26623376623365),
    (14.876623376623343, 461.42207792207785),
    (5.525974025973994, 570.512987012987),
    (530.7207792207791, 568.9545454545454),
    (660.0714285714286, 137.26623376623365),
    (552.538961038961, 87.3961038961038)
]

# Define regions
regions = {
    "road/way": [np.array(road_way_coords)],
    "parking": [np.array(parking_1_coords), np.array(parking_2_coords)]
}

# Global dictionary to store counts of vehicles in each region
region_counts = {
    "road/way": 0,
    "parking": 0
    
}

def is_within_regions(bbox, regions, threshold=0.6, sampling_density=10):
    total_points = sampling_density * sampling_density
    for region_name, region_coords in regions.items():
        for region in region_coords:
            int_region = np.array(region, dtype=np.int32)

            # Sample points within the bounding box
            x_points = np.linspace(bbox[0], bbox[2], sampling_density)
            y_points = np.linspace(bbox[1], bbox[3], sampling_density)
            inside_count = 0

            for x in x_points:
                for y in y_points:
                    if cv2.pointPolygonTest(int_region, (x, y), False) >= 0:
                        inside_count += 1

            # Check if the percentage of points inside the region meets the threshold
            if inside_count / total_points >= threshold:
                return True
    return False


def count_vehicles_in_regions(boxes, regions):
    counts = {region_name: 0 for region_name in regions.keys()}

    for box in boxes:
        max_intersection_area = 0
        region_for_box = None

        box_polygon = np.array([
            [box[0], box[1]],
            [box[0], box[3]],
            [box[2], box[3]],
            [box[2], box[1]]
        ], dtype=np.int32)

        for region_name, region_coords in regions.items():
            for region in region_coords:
                region_np = np.array(region, dtype=np.int32)
                results = cv2.intersectConvexConvex(box_polygon, region_np)
                
                # Check the length of the results to determine what was returned
                if len(results) == 2:
                    ret, intersect_area = results
                else:
                    ret, intersect_area, _ = results
                
                if ret and np.sum(intersect_area) > max_intersection_area:
                    max_intersection_area = np.sum(intersect_area)
                    region_for_box = region_name

        if region_for_box:
            counts[region_for_box] += 1

    return counts


def initialize_gpu():
    torch.cuda.empty_cache()
    torch.cuda.init()

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video/RTSP stream")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return cap, frame_width, frame_height, fps


def create_video_writer(fps, frame_width, frame_height):
    return cv2.VideoWriter(f"output_video/{time.time()}.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

def optimize_for_ocr(plate_img):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using histogram equalization
    contrast_enhanced = cv2.equalizeHist(gray)
    
    # Global Thresholding
    _, binary = cv2.threshold(contrast_enhanced, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Resize while maintaining aspect ratio
    aspect_ratio = plate_img.shape[1] / plate_img.shape[0]
    resized_width = int(100 * aspect_ratio)
    standardized = cv2.resize(binary, (resized_width, 100))
    
    return standardized

def extract_lp_data(data):
    if isinstance(data, list):
        for item in data:
            text, conf = extract_lp_data(item)
            if text and conf:
                return text, conf
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], str) and isinstance(data[1], float):
        return ''.join([char for char in data[0] if char in valid_chars]), round(data[1], 2)
    return None, None

def process_license_plate_paddle_ocr(frame, xmin, ymin, xmax, ymax):
    plate = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

    # Optimize the plate image for OCR
    #optimized_plate = optimize_for_ocr(plate)

    image = Image.fromarray(plate)

    temp_image_path = 'temp/temp_plate_image.jpg'
    
    image.save(temp_image_path)

    # Recognize text from the image
    result = paddle_ocr.ocr(temp_image_path)  # Set use_gpu=True if you have a GPU and want to use it

    lp_text, lp_conf = extract_lp_data(result)
    return lp_text, lp_conf, plate

def process_license_plate_pytesseract(frame, xmin, ymin, xmax, ymax):
    plate = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    # Optimize the plate image for OCR
    optimized_plate = optimize_for_ocr(plate)

    # Perform OCR with Tesseract
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(optimized_plate, config=custom_config, output_type=pytesseract.Output.DICT)

    lp_text = None
    lp_conf = None

    # Handle hyphenated plates
    has_hyphens = any('-' in t for t in data['text'])
    if has_hyphens:
        lp_text = max(data['text'], key=len)
        lp_conf = data['conf'][data['text'].index(lp_text)]
        dummy_txt = lp_text.upper()
        lp_text = dummy_txt.strip()

        if len(lp_text) < 5 or any(c not in valid_chars for c in lp_text):
            return None, None, optimized_plate

    # Handle non-hyphenated plates
    else:
        lp_texts = []
        for i in range(len(data['text'])):
            text = data['text'][i]
            if text.strip() and data['conf'][i] > 25:
                lp_texts.append(text)

        if len(lp_texts) == 0:
            return None, None, optimized_plate

        lp_text = ''.join(lp_texts).upper()
        if len(lp_text) < 5 or any(c not in valid_chars for c in lp_text):
            return None, None, optimized_plate

        lp_conf = sum(data['conf'][data['text'].index(t)] for t in lp_texts) / len(lp_texts)

    return lp_text, lp_conf, optimized_plate  # Added lp_resized to the return statement

def process_license_plate_easyocr(frame, xmin, ymin, xmax, ymax):
    plate = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    #cv2.imwrite(f'temp/{time.time()}.jpg', plate)
    
    # Optimize the plate image for OCR
    optimized_plate = optimize_for_ocr(plate)

    results  = reader.readtext(optimized_plate)
    #print(results)
    if results:
        result = results[0]
        lp_text = result[1]
        lp_conf = result[2]

        return lp_text, lp_conf, optimized_plate  # Added lp_resized to the return statement

    else:
        return None, None, optimized_plate
    
def process_license_plate_openalpr(frame, xmin, ymin, xmax, ymax):
    plate = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    # Optimize the plate image for OCR
    optimized_plate = optimize_for_ocr(plate)
    
    # Convert image to bytes
    is_success, im_buf_arr = cv2.imencode(".jpg", optimized_plate)
    byte_im = im_buf_arr.tobytes()

    # Process with "us" configuration
    result_us = subprocess.check_output(["alpr", "-c", "us", "-"], input=byte_im)
    lp_text_us, lp_conf_us, _ = parse_alpr_result(result_us)

    # Process with "eu" configuration
    result_eu = subprocess.check_output(["alpr", "-c", "eu", "-"], input=byte_im)
    lp_text_eu, lp_conf_eu, _ = parse_alpr_result(result_eu)

    # Handling the scenarios
    if lp_conf_us is None and lp_conf_eu is None:
        return None, None, plate
    elif lp_conf_us is None:
        return lp_text_eu, lp_conf_eu, plate
    elif lp_conf_eu is None:
        return lp_text_us, lp_conf_us, plate
    else:
        # Compare and return the result with the highest confidence
        if lp_conf_us > lp_conf_eu:
            return lp_text_us, lp_conf_us, plate
        else:
            return lp_text_eu, lp_conf_eu, plate

def parse_alpr_result(result):
    lines = result.decode('utf-8').split("\n")
    for idx, line in enumerate(lines):
        if "plate" in line:
            # Ensure that we don't go out of bounds
            if idx + 1 < len(lines):
                parts = lines[idx + 1].strip().split("\t")
                if len(parts) > 1:
                    lp_text = parts[0].replace("-", "").strip()  # Extracting the license plate text and cleaning
                    lp_conf = float(parts[1].split(":")[1].strip()) / 100.0  # Extracting the confidence value
                    return lp_text, lp_conf, None
    return None, None, None

def are_similar(str1, str2, threshold=0.7):
    """
    Check if two license plates are potentially the same.
    """
    s = difflib.SequenceMatcher(None, str1, str2)
    ratio = s.ratio()
    #print(f"Comparing '{str1}' and '{str2}' gives a similarity ratio of: {ratio}")
    return s.ratio() > threshold


def resize_frame(frame, width=640):
    aspect_ratio = frame.shape[1] / float(frame.shape[0])
    height = int(width / aspect_ratio)
    return cv2.resize(frame, (width, height))

def process_frame(frame, frame_num, car_record, frames_record, backframes_record, model, video_writer, timestamp, ocr_choice, is_rtsp=False):
    
    global total_detected, successful_ocr, unsuccessful_ocr, count_in_parking, count_in_roadway, total_detected_wo_rtsp
    id = None

    # Reset the vehicle counts for the regions
    region_counts = {
        "road/way": 0,
        "parking": 0
        
    }

    # Dictionary for region colors
    region_colors = {
        "road/way": (0, 0, 255),  # Red
        "parking": (0, 255, 0)  # Green
        
    }

    if is_rtsp:
        # Draw regions on the frame
        for idx, (region_name, region_coords) in enumerate(regions.items()):
            for region in region_coords:
                int_region = np.array(region, dtype=np.int32)
                color = region_colors.get(region_name, (0, 255, 0))  # Default to green if region name is not recognized
                cv2.polylines(frame, [int_region], isClosed=True, color=color, thickness=2)
                
                # Label the region. Draw a filled rectangle for the background.
                text_size = cv2.getTextSize(region_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Reposition the labels to the top-left corner of the region
                text_position = (int_region[0][0], int_region[0][1] - 10 - idx*30)  # position slightly above the first point of the region and adjust based on the region index
                rect_top_left = (text_position[0] - 5, text_position[1] - text_size[1] - 5)
                rect_bottom_right = (text_position[0] + text_size[0] + 5, text_position[1] + 5)
                
                cv2.rectangle(frame, rect_top_left, rect_bottom_right, color, -1)  # -1 thickness makes the rectangle filled
                cv2.putText(frame, region_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    frame = resize_frame(frame)
    height, width = frame.shape[:2]


    vehicle_results = model_vehicle.track(frame, persist=True, classes=[2,3,5,7], conf=0.30)

    # Dictionary to map vehicle IDs to their bounding boxes
    vehicle_bbox_dict = {}
    det_conf_threshold = app.config['DET_CONF']

    # Check if vehicles are detected
    if vehicle_results[0].boxes.id is not None:
        vehicle_boxes = vehicle_results[0].boxes.xyxy.tolist()
        vehicle_ids = vehicle_results[0].boxes.id.int().tolist()
        vehicle_confidences = vehicle_results[0].boxes.conf.tolist()


        # Dictionary to map vehicle IDs to their bounding boxes
        vehicle_bbox_dict = {vehicle_ids[i]: vehicle_boxes[i] for i in range(len(vehicle_ids))}
        
        # Draw each vehicle's bounding box on the frame
        for  i, (vehicle_id, (v_xmin, v_ymin, v_xmax, v_ymax)) in enumerate(vehicle_bbox_dict.items()):
            if vehicle_confidences[i] < det_conf_threshold:
                continue  # Skip this detection if it's below the threshold

            v_xmin, v_ymin, v_xmax, v_ymax = [int(v) for v in [v_xmin, v_ymin, v_xmax, v_ymax]]
            v_xmin, v_ymin = max(0, v_xmin), max(0, v_ymin)
            v_xmax, v_ymax = min(width, v_xmax), min(height, v_ymax)

            # Check if the vehicle is within any of the defined regions
            # if not is_within_regions([v_xmin, v_ymin, v_xmax, v_ymax], regions):
            #    continue

            if is_rtsp:
                region_resides = count_vehicles_in_regions([(v_xmin, v_ymin, v_xmax, v_ymax)], regions)
                for region, count in region_resides.items():
                    region_counts[region] += count
                
                if vehicle_id not in counted_vehicle_ids:  # Only count if the vehicle hasn't been counted before
                    region_resides = count_vehicles_in_regions([(v_xmin, v_ymin, v_xmax, v_ymax)], regions)
                    for region, count in region_resides.items():
                        region_counts[region] += count

                    # New code to update the global counts
                    if region_resides.get("parking", 0) > 0:
                        count_in_parking += 1
                    elif region_resides.get("road/way", 0) > 0:
                        count_in_roadway += 1
                    total_detected += 1
                    

                    counted_vehicle_ids.add(vehicle_id)  # Mark the vehicle as counted
            
            else:
                    if vehicle_id not in counted_vehicle_ids:
                        total_detected_wo_rtsp += 1
                        counted_vehicle_ids.add(vehicle_id)

            cv2.rectangle(frame, (v_xmin, v_ymin), (v_xmax, v_ymax), (255, 255, 0), 2)  # Cyan color for vehicle boxes

    results = model.predict(frame, device=0)
    det_conf_threshold = app.config['DET_CONF']
    print(f"detection confidence threshold is: {det_conf_threshold}")
    current_frame_ids = set()

    OCR_FUNCTIONS = {
        '1': process_license_plate_easyocr,
        '2': process_license_plate_openalpr,
        '3': process_license_plate_paddle_ocr,
        '4': process_license_plate_pytesseract
    }

    OCR_ENGINES = {
        '1': 'EasyOCR',
        '2': 'OpenALPR',
        '3': 'PaddleOCR',
        '4': 'Pytesseract'
    }

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.tolist()
        confidences = results[0].boxes.conf.tolist()
        #track_ids = results[0].boxes.id.int().tolist()
        
        
        ocr_method = OCR_ENGINES[ocr_choice]
        conf_threshold = app.config['OCR_CONF']
        det_conf_threshold = app.config['DET_CONF']

        print(f"confidence threshold is: {conf_threshold}")

        for i, box in enumerate(boxes):
            if confidences[i] >= det_conf_threshold:
                
                lp_text = None
                lp_conf = None
                
                x1, y1, x2, y2 = [int(v) for v in boxes[i]]
                xmin, ymin = max(0, x1), max(0, y1)
                xmax, ymax = min(width, x2), min(height, y2)

                # Check if the license plate is within any of the defined regions
                if not is_within_regions([xmin, ymin, xmax, ymax], regions):
                    continue

                # Check if the detected license plate is inside any vehicle bounding box
                for vehicle_id, (vehicle_xmin, vehicle_ymin, vehicle_xmax, vehicle_ymax) in vehicle_bbox_dict.items():
                    if (xmin >= vehicle_xmin and xmax <= vehicle_xmax and
                        ymin >= vehicle_ymin and ymax <= vehicle_ymax):
                        
                        id = vehicle_id
                        
                        break  # Exit the loop once a match is found

                if id is not None:
                    #backframes_record.setdefault(id, {'id': id, 'box': [xmin, ymin, xmax, ymax], 'first_frame': frame_num})
                    current_frame_ids.add(id)

                    if frame_num % 3 == 0:
                        # Run the OCR function first to update lp_text and lp_conf
                        lp_text, lp_conf, lp_resized = OCR_FUNCTIONS[ocr_choice](frame, xmin, ymin, xmax, ymax)
                        
                        # Check the condition to update car_record only if lp_conf is not None
                        if lp_conf is not None and (id not in car_record or lp_conf > car_record[id]['confidence']):
                            if lp_text and lp_conf >= conf_threshold and len(lp_text) > 3:
                                lp_text = lp_text.upper()
                                ts_str = timestamp
                                # Check if the license plate text is already in the live_data
                                existing_ids = [key for key, data in live_data.items() if data['ocr_reading'] == lp_text]
                                
                                lp_image = Image.fromarray(lp_resized)
                                
                                SAVE_PATH = os.path.join('static', 'license_plates')
                                lp_image_name = f'{time.time()}.jpg'
                                lp_image_path = os.path.join(SAVE_PATH, lp_image_name)

                                lp_image.save(lp_image_path)
                                
                                if not existing_ids:
                                    # If the license plate text doesn't exist, add the new entry
                                    live_data[id] = {
                                        'track_id': id,
                                        'plate_image_url': lp_image_path,
                                        'ocr_reading': lp_text,
                                        'time_stamp': timestamp,
                                        'confidence': lp_conf,
                                        'ocr_engine': ocr_method
                                    }
                                else:
                                    # If it exists, compare the confidence values and keep the one with the higher confidence
                                    existing_id = existing_ids[0]
                                    if lp_conf > live_data[existing_id]['confidence']:
                                        live_data[existing_id] = {
                                            'track_id': existing_id,
                                            'plate_image_url': lp_image_path,
                                            'ocr_reading': lp_text,
                                            'time_stamp': timestamp,
                                            'confidence': lp_conf,
                                            'ocr_engine': ocr_method
                                        }
                                    
                                
                                # Check for previous detection of the same or similar license plate
                                matching_ids = [k for k, data in car_record.items() if are_similar(data['text'], lp_text)]

                                if matching_ids:
                                    matching_id = matching_ids[0]
                                    prev_data = car_record[matching_id]

                                    # Check spatial proximity
                                    intersection_area = (min(xmax, prev_data['box'][2]) - max(xmin, prev_data['box'][0])) * \
                                                        (min(ymax, prev_data['box'][3]) - max(ymin, prev_data['box'][1]))
                                    union_area = (xmax - xmin) * (ymax - ymin) + \
                                                (prev_data['box'][2] - prev_data['box'][0]) * (prev_data['box'][3] - prev_data['box'][1]) - \
                                                intersection_area
                                    iou = intersection_area / union_area

                                    # Check time difference
                                    ts_current = datetime.strptime(ts_str, "%H:%M:%S")
                                    ts_previous = datetime.strptime(prev_data['timestamp'], "%H:%M:%S")
                                    time_diff = (ts_current - ts_previous).total_seconds()

                                    # If spatially close and temporally close, and new confidence is significantly better, update
                                    if iou > 0.7 and time_diff < 2 and lp_conf > prev_data['confidence'] * 1.05:
                                        car_record[matching_id] = {
                                            'text': lp_text,
                                            'box': [xmin, ymin, xmax, ymax],
                                            'timestamp': ts_str,
                                            'confidence': lp_conf
                                        }
                                    else:
                                        car_record[id] = {
                                            'text': lp_text,
                                            'box': [xmin, ymin, xmax, ymax],
                                            'timestamp': ts_str,
                                            'confidence': lp_conf
                                        }
                                else:
                                    car_record[id] = {
                                        'text': lp_text,
                                        'box': [xmin, ymin, xmax, ymax],
                                        'timestamp': ts_str,
                                        'confidence': lp_conf
                                    }
                                    
                                frames_record[id] = {'text': lp_text, 'box': [xmin, ymin, xmax, ymax], 'timestamp': timestamp, 'confidence': lp_conf}
                                # If OCR was successful, increment the appropriate successful_ocr counter
                                if lp_text and id not in successful_ocr_ids:
                                    successful_ocr += 1
                                successful_ocr_ids.add(id)
                                #cv2.imwrite(f'out/{time.time()}.jpg', lp_resized)

                # Check if the id is not in processed_ids before updating unsuccessful_ocr
                if not is_rtsp:
                    unsuccessful_ocr = total_detected_wo_rtsp - successful_ocr
                    processed_ids.add(id)
                elif is_rtsp:
                    unsuccessful_ocr = total_detected - successful_ocr
                    processed_ids.add(id)

        video_writer.write(frame)
    else:
        video_writer.write(frame)
          
    #total_detected = total_detected + successful_ocr + unsuccessful_ocr + count_in_parking + count_in_roadway
    write_records_to_csv(car_record)  # Write to CSV after every frame is processed
    return frame

def write_records_to_csv(car_record):
    # Ensure the output directory exists
    if not os.path.exists('record'):
        os.makedirs('record')

    # Read the existing CSV into memory
    existing_data = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[int(row['track_id'])] = row

    # Update the records
    for id, data in car_record.items():
        existing_data[id] = {
            'track_id': id,
            'license_plate': data['text'],
            'bounding_box': data['box'],
            'timestamp': data['timestamp'],
            'confidence': data['confidence']
        }

    # Write the updated records back to the CSV
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = ['track_id', 'license_plate', 'bounding_box', 'timestamp', 'confidence']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for id, data in existing_data.items():
            writer.writerow(data)

def video_stream(input_type, path, model, ocr_choice):
    
    # Ensure the output directory exists
    if not os.path.exists('output_video'):
        os.makedirs('output_video')
    
    if not os.path.exists('record'):
        os.makedirs('record')

    frame = None

    os.makedirs('temp', exist_ok=True)

    global current_ocr_choice

    if input_type == 'video' or input_type == 'rtsp':
        cap, frame_width, frame_height, fps = open_video(path)
        if not cap.isOpened():
            flash("Error opening video stream")
            return redirect(url_for('index'))
        video_writer = create_video_writer(fps, frame_width, frame_height)
    
    elif input_type == 'directory':
        images = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        image_iter = iter(images)
        video_writer = None  # Initialize video_writer to None
    
    else:  # For individual images
        frame = cv2.imread(path)
        frame_height, frame_width = frame.shape[:2]
        fps = 1
        video_writer = create_video_writer(fps, frame_width, frame_height)

    car_record = {}
    frames_record = {}
    backframes_record = {}
    frame_num = 0

    while True:
        ts_str = None  # Initialize timestamp to None


        # Check the current OCR choice
        if current_ocr_choice:
            ocr_method = current_ocr_choice
        else:
            ocr_method = ocr_choice


        if input_type == 'video' or input_type == 'rtsp':
            ret, frame = cap.read()
            if not ret:
                break

            # Extract timestamp for the video
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts_mins, ts_secs = divmod(timestamp / 1000, 60)  # Convert to minutes and seconds
            ts_hours, ts_mins = divmod(ts_mins, 60)          # Convert to hours and minutes
            ts_str = f"{int(ts_hours):02}:{int(ts_mins):02}:{int(ts_secs):02}" # Format as hh:mm:ss

            # Check for negative timestamp values
            if ts_hours < 0 or ts_mins < 0 or ts_secs < 0:
                ts_str = '00:00:00'
            else:
                ts_str = f"{int(ts_hours):02}:{int(ts_mins):02}:{int(ts_secs):02}" # Format as hh:mm:ss
        
        elif input_type == 'directory':
            try:
                frame = cv2.imread(next(image_iter))
                ts_str = datetime.now().strftime("%H:%M:%S")  # Use the current time as a placeholder since we're not processing a video

                if video_writer is None:  # First image, create video_writer
                    frame_height, frame_width = frame.shape[:2]
                    fps = 1  # Assuming 1 frame per second for images
                    video_writer = create_video_writer(fps, frame_width, frame_height)
            except StopIteration:
                break
        
        

        # Check if frame is None (e.g., no images in directory)
        if frame is None:
            break

        frame = process_frame(frame, frame_num, car_record, frames_record, backframes_record, model, video_writer, ts_str, ocr_method, is_rtsp=(input_type == 'rtsp'))
        frame_num += 1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n\r\n')
        
        if input_type == 'image':
            break
        
        write_records_to_csv(car_record)

    if video_writer:
        video_writer.release()
    
    write_records_to_csv(car_record)
    
    shutil.rmtree('temp')

@app.route('/get_counts')
def get_counts():
    input_type = session.get('input_type')
    if input_type == 'rtsp':
        return jsonify({
            'total_detected': total_detected,
            'count_in_parking': count_in_parking,
            'count_in_roadway': count_in_roadway,
            'successful_ocr': successful_ocr,
            'unsuccessful_ocr': unsuccessful_ocr
        })
    else:
        return jsonify({
            'total_detected': total_detected_wo_rtsp,
            'successful_ocr': successful_ocr,
            'unsuccessful_ocr': unsuccessful_ocr
        })

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    input_type = session.get('input_type')
    path = session.get('path')
    ocr_choice = session.get('ocr_choice')
    return Response(stream_with_context(video_stream(input_type, path, model, ocr_choice)),  # Using Flask's stream_with_context for better streaming performance
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_confidences', methods=['POST'])
def update_confidences():
    data = request.get_json()
    app.config['OCR_CONF'] = float(data['ocrConf'])
    app.config['DET_CONF'] = float(data['detConf'])

    print(f"Updated confidence values. OCR: {ocr_conf_global}, Detection: {det_conf_global}")
    return jsonify(success=True)

@app.route('/get_live_data', methods=['GET'])
def get_live_data():
    return jsonify(live_data)

@app.route('/change_ocr', methods=['POST'])
def change_ocr():
    global current_ocr_choice
    data = request.get_json()
    new_choice = data.get('newChoice')
    
    # Update the global variable with the new choice
    current_ocr_choice = new_choice
    
    return jsonify(status='success')

@app.route('/reset_and_home', methods=['GET'])
def reset_and_home():
    clear_global_variables()
    session.clear()  # Clear the session data
    
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    global ocr_conf_global
    if request.method == 'POST':
        input_type = request.form['inputType']
        ocr_choice = request.form['ocrMethod']

        # Update the global ocr_conf_global variable
        ocr_conf_global = 0.50 if ocr_choice in ["1", "4"] else 0.90 if ocr_choice == "3" else 0.80 if ocr_choice == "2" else 0.50
        app.config['OCR_CONF'] = ocr_conf_global

        if input_type != 'rtsp':
            if 'file' not in request.files:
                flash("No file uploaded")
                return redirect(url_for('index'))
            
            file = request.files['file']
            if file.filename == '':
                flash("No file selected")
                return redirect(url_for('index'))
            
            if file and allowed_file(file.filename):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                session['input_type'] = input_type
                session['path'] = filename
                session['ocr_choice'] = ocr_choice
                return render_template('result.html', ocr_conf=app.config['OCR_CONF'], ocr_choice=ocr_choice, input_type=input_type)

        else:
            rtsp_url = request.form['rtspUrl']
            session['input_type'] = 'rtsp'
            session['path'] = rtsp_url
            session['ocr_choice'] = ocr_choice
            return render_template('result.html', ocr_conf=app.config['OCR_CONF'], ocr_choice=ocr_choice, input_type=input_type)


        flash("Processing completed successfully!")
        return redirect(url_for('index'))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)