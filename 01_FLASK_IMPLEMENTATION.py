# ============================================
# TRAFFIC VISION AI - FLASK BACKEND
# ============================================
from Config import FlaskConfig, YOLOConfig, TrafficControlConfig, VideoProcessingConfig
from flask import Flask, render_template, request, jsonify, session, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path
import json
import threading
from collections import defaultdict

# ===== CONFIGURATION =====
app = Flask(__name__)
app.config.from_object(FlaskConfig) # 500MB max file size

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# ===== DATABASE MODELS =====
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='admin')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TrafficSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_date = db.Column(db.DateTime, default=datetime.utcnow)
    total_vehicles = db.Column(db.Integer, default=0)
    total_duration = db.Column(db.Float, default=0)
    
class LaneData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('traffic_session.id'))
    lane_number = db.Column(db.Integer)
    car_count = db.Column(db.Integer, default=0)
    bus_count = db.Column(db.Integer, default=0)
    truck_count = db.Column(db.Integer, default=0)
    motorcycle_count = db.Column(db.Integer, default=0)
    total_vehicles = db.Column(db.Integer, default=0)
    signal_duration = db.Column(db.Float, default=0)
    ambulance_detected = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ===== YOLO MODEL LOADER =====
class TrafficDetector:
    def __init__(self):
        # Load vehicle detection model
        self.vehicle_model = YOLO(YOLOConfig.VEHICLE_MODEL)  # nano model for speed
        # For ambulance detection, use a fine-tuned model if available
        # self.ambulance_model = YOLO('ambulance_model.pt')
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
    def detect_vehicles(self, frame):
        """Detect vehicles in frame"""
        results = self.vehicle_model(frame)
        detections = {
            'car': 0,
            'motorcycle': 0,
            'bus': 0,
            'truck': 0
        }
        boxes = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence > 0.5:  # confidence threshold
                    if class_id in self.vehicle_classes:
                        vehicle_type = self.vehicle_classes[class_id]
                        detections[vehicle_type] += 1
                        
                        # Get bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append({
                            'type': vehicle_type,
                            'coords': (x1, y1, x2, y2),
                            'confidence': confidence
                        })
        
        return detections, boxes
    
    def detect_ambulance(self, frame):
        """Detect ambulance - can be replaced with custom model"""
        # Simple heuristic: look for blue+white patterns
        # In production, use fine-tuned YOLO model
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue color range (ambulance typically blue)
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # If significant blue area detected, likely ambulance
        return len(contours) > 0 and cv2.countNonZero(mask) > 1000

# ===== TRAFFIC LOGIC =====
class TrafficController:
    def __init__(self):
        self.base_time = TrafficControlConfig.BASE_TIME  # base green light duration
        self.max_time = 60
        self.min_time = 5
        self.ambulance_time = 15  # priority time for ambulance
        
    def calculate_signal_duration(self, vehicle_count):
        """Calculate green signal duration based on vehicle density"""
        if vehicle_count == 0:
            return self.min_time
        
        # Formula: base_time + (vehicle_count * 0.5 seconds per vehicle)
        duration = self.base_time + (vehicle_count * 0.5)
        return min(duration, self.max_time)
    
    def get_lane_priority(self, lane_densities):
        """Get priority order based on vehicle density"""
        # Sort lanes by density (descending)
        sorted_lanes = sorted(
            enumerate(lane_densities),
            key=lambda x: x[1]['total'],
            reverse=True
        )
        return sorted_lanes

# ===== VIDEO PROCESSING =====
class VideoProcessor:
    def __init__(self):
        self.detector = TrafficDetector()
        self.controller = TrafficController()
        self.frame_skip = VideoProcessingConfig.FRAME_SKIP
  # Process every 5th frame
        
    def process_video(self, video_path, session_id):
        """Process video and generate traffic simulation"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        lane_statistics = {'car': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0}
        
        ambulance_detected = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame
            if frame_count % self.frame_skip == 0:
                # Detect vehicles
                detections, boxes = self.detector.detect_vehicles(frame)
                
                # Detect ambulance
                if self.detector.detect_ambulance(frame):
                    ambulance_detected = True
                
                # Draw on frame (optional)
                for box in boxes:
                    x1, y1, x2, y2 = box['coords']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, box['type'], (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Update statistics
                for vtype, count in detections.items():
                    lane_statistics[vtype] += count
        
        cap.release()
        
        return {
            'car': lane_statistics['car'],
            'bus': lane_statistics['bus'],
            'truck': lane_statistics['truck'],
            'motorcycle': lane_statistics['motorcycle'],
            'total': sum([lane_statistics[key] for key in lane_statistics]),
            'ambulance_detected': ambulance_detected
        }

# ===== ROUTES =====

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(username=data['username']).first()
        
        if user and check_password_hash(user.password, data['password']):
            session['user_id'] = user.id
            return jsonify({'success': True, 'message': 'Login successful'})
        
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('frontend.html')
# Background video processing implemented in the later function.

@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True})

# Dashboard Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('frontend.html')

@app.route('/features')
def features():
    return render_template('frontend.html')

@app.route('/start')
def start():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('frontend.html')

# API Routes
@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    try:
        files = request.files
        lane_videos = {}
        
        # Handle 4 lanes
        for i in range(1, 5):
            lane_key = f'lane{i}_video'
            if lane_key in files:
                file = files[lane_key]
                filename = secure_filename(f'lane_{i}_{datetime.now().timestamp()}.mp4')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                lane_videos[f'lane{i}'] = filepath
        
        # Create session
        session_obj = TrafficSession()
        db.session.add(session_obj)
        db.session.commit()
       
        # Process videos in background
        threading.Thread(
            target=process_videos_background,
            args=(lane_videos, session_obj.id)
        ).start()
        
        return jsonify({
            'success': True,
            'session_id': session_obj.id,
            'message': 'Videos uploaded. Processing started...'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new traffic session"""
    try:
        session_obj = TrafficSession()
        db.session.add(session_obj)
        db.session.commit()
        
        return jsonify({
            'session_id': session_obj.id,
            'session_date': session_obj.session_date.isoformat(),
            'total_vehicles': 0,
            'status': 'initialized'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/sessions/<int:session_id>/upload', methods=['POST'])
def upload_session_videos(session_id):
    """Upload videos for a session"""
    try:
        session_obj = TrafficSession.query.get(session_id)
        if not session_obj:
            return jsonify({'error': 'Session not found'}), 404
        
        files = request.files
        lane_videos = {}
        
        # Handle 4 lanes
        for i in range(1, 5):
            lane_key = f'lane{i}_video'
            if lane_key in files:
                file = files[lane_key]
                filename = secure_filename(f'lane_{i}_{session_id}_{datetime.now().timestamp()}.mp4')
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                lane_videos[f'lane{i}'] = (filepath, i)
        
        # Process videos in background
        threading.Thread(
            target=process_videos_background,
            args=(lane_videos, session_id)
        ).start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Videos uploaded. Processing started...'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/api/sessions/<int:session_id>/status')
def get_session_status(session_id):
    """Get session processing status"""
    try:
        session_obj = TrafficSession.query.get(session_id)
        if not session_obj:
            return jsonify({'error': 'Session not found'}), 404
        
        lanes = LaneData.query.filter_by(session_id=session_id).all()
        
        return jsonify({
            'session_id': session_id,
            'total_vehicles': session_obj.total_vehicles,
            'lanes': [{
                'lane_number': lane.lane_number,
                'car_count': lane.car_count,
                'bus_count': lane.bus_count,
                'truck_count': lane.truck_count,
                'motorcycle_count': lane.motorcycle_count,
                'total_vehicles': lane.total_vehicles,
                'signal_duration': lane.signal_duration,
                'ambulance_detected': lane.ambulance_detected
            } for lane in lanes]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/sessions/<int:session_id>/analysis')
def get_analysis(session_id):
    """Get traffic analysis for a session"""
    try:
        lanes = LaneData.query.filter_by(session_id=session_id).all()
        
        if not lanes:
            return jsonify({'error': 'No data available'}), 404
        
        analysis = {
            'lane_densities': [],
            'vehicle_breakdown': {},
            'signal_timings': [],
            'ambulance_events': 0,
            'traffic_reduction_percentage': 0
        }
        
        total_cars = total_buses = total_trucks = total_motorcycles = 0
        total_reduction = 0
        
        for lane in lanes:
            analysis['lane_densities'].append({
                'lane': f'Lane {lane.lane_number}',
                'density': lane.total_vehicles
            })
            
            reduction = ((30 - lane.signal_duration) / 30) * 100
            analysis['signal_timings'].append({
                'lane': f'Lane {lane.lane_number}',
                'traditional': 30,
                'proposed': lane.signal_duration,
                'reduction': round(reduction, 2)
            })
            total_reduction += reduction
            
            if lane.ambulance_detected:
                analysis['ambulance_events'] += 1
            
            total_cars += lane.car_count
            total_buses += lane.bus_count
            total_trucks += lane.truck_count
            total_motorcycles += lane.motorcycle_count
        
        analysis['vehicle_breakdown'] = {
            'cars': total_cars,
            'buses': total_buses,
            'trucks': total_trucks,
            'motorcycles': total_motorcycles
        }
        
        if lanes:
            analysis['traffic_reduction_percentage'] = round(total_reduction / len(lanes), 2)
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def process_videos_background(lane_videos, session_id):
    """Process videos in background thread"""
    with app.app_context():
        processor = VideoProcessor()
        
        try:
            for lane_key, video_info in lane_videos.items():
                if isinstance(video_info, tuple):
                    video_path, lane_num = video_info
                else:
                    video_path = video_info
                    lane_num = int(lane_key[-1])
                
                stats = processor.process_video(video_path, session_id)
                
                if stats:
                    # Calculate signal duration
                    total_vehicles = stats['total']
                    signal_duration = processor.controller.calculate_signal_duration(total_vehicles)
                    
                    # Store in database
                    lane_data = LaneData(
                        session_id=session_id,
                        lane_number=lane_num,
                        car_count=stats['car'],
                        bus_count=stats['bus'],
                        truck_count=stats['truck'],
                        motorcycle_count=stats['motorcycle'],
                        total_vehicles=total_vehicles,
                        signal_duration=signal_duration,
                        ambulance_detected=stats['ambulance_detected']
                    )
                    db.session.add(lane_data)
                    
                    # Update session total
                    session_obj = TrafficSession.query.get(session_id)
                    session_obj.total_vehicles += total_vehicles
            
            db.session.commit()
            
        except Exception as e:
            print(f"Error processing videos: {e}")
            db.session.rollback()

# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# ===== INITIALIZATION =====
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create default admin user if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                password=generate_password_hash('admin_password'),
                role='admin'
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created: admin / admin_password")
        
        print("\n" + "="*60)
        print("🚦 TRAFFIC VISION AI - FLASK SERVER STARTING")
        print("="*60)
        print("✅ Database initialized")
        print("✅ Models loaded")
        print("📍 Server: http://127.0.0.1:5000")
        print("👤 Login: admin / admin_password")
        print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
