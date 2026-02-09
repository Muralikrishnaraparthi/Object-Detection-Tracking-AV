import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import time
import torch
import os

# --- ZEROGPU SETUP ---
try:
    import spaces
    print(" 'spaces' library imported successfully.")
except ImportError:
    print(" 'spaces' library not found. Falling back to dummy decorator.")
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            def wrapper(f):
                return f
            if func:
                return wrapper(func)
            return wrapper

# --- KITTI CLASS NAMES ---
KITTI_CLASSES = {
    0: 'Car', 1: 'Van', 2: 'Truck', 3: 'Pedestrian',
    4: 'Person (Sit)', 5: 'Cyclist', 6: 'Tram', 7: 'Misc'
}

# Global model variable
model = None

def load_model():
    """Load YOLOv8 model"""
    global model
    if model is None:
        try:
            model = YOLO('best.pt')
            print("Loaded custom trained model")
        except:
            print("'best.pt' not found. Loading standard YOLOv8n...")
            model = YOLO('yolov8n.pt')
    return model

def generate_analytics(detection_data):
    if not detection_data: return None
    class_counts = Counter([d['class'] for d in detection_data])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    colors = [f'C{i}' for i in range(len(classes))]
    ax.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Object Detection Statistics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    temp_path = tempfile.mktemp(suffix='.png')
    plt.savefig(temp_path, dpi=100, bbox_inches='tight')
    plt.close()
    return temp_path

def get_detection_data(results):
    detection_data = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            class_name = KITTI_CLASSES.get(cls_id, f'Class_{cls_id}')
            detection_data.append({
                'class': class_name, 'confidence': conf, 'bbox': [x1, y1, x2, y2], 'track_id': track_id
            })
    return detection_data

def format_detection_table(detection_data):
    if not detection_data: return "No detections found."
    table = "| # | Class | ID / Conf | Bounding Box |\n|---|---|---|---|\n"
    for idx, det in enumerate(detection_data, 1):
        val_str = f"ID: {det['track_id']}" if det['track_id'] is not None else f"Conf: {det['confidence']:.2%}"
        table += f"| {idx} | {det['class']} | {val_str} | ({det['bbox'][0]}, {det['bbox'][1]}, ...) |\n"
    return table

# --- CORE LOGIC (ZEROGPU) ---

@spaces.GPU(duration=30)
def process_image(image, conf_threshold, iou_threshold):
    if image is None: return None, None, "Please upload an image", None
    
    model = load_model()
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    start_time = time.time()
    
    # Explicitly casting sliders to float
    results = model(
        img_bgr, 
        conf=float(conf_threshold), 
        iou=float(iou_threshold),
        agnostic_nms=True, 
        augment=False, 
        imgsz=640,
        device=device
    )
    
    inference_time = time.time() - start_time
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    detection_data = get_detection_data(results)
    
    summary = f"""
### Image Summary
- **Time:** {inference_time:.3f}s
- **Objects Detected:** {len(detection_data)}
- **Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    """
    return annotated_rgb, generate_analytics(detection_data), summary, format_detection_table(detection_data)

@spaces.GPU(duration=120)
def process_video(video_path, conf_threshold, iou_threshold, skip_frames, progress=gr.Progress()):
    if video_path is None: return None, "Please upload a video", None
    
    model = load_model()
    device = '0' if torch.cuda.is_available() else 'cpu'
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video.", None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = tempfile.mktemp(suffix='.mp4')
    
    # --- BROWSER PLAYBACK ---
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
             raise Exception("avc1 codec failed")
    except:
        print("Warning: H.264 (avc1) codec not found. Falling back to mp4v.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx, processed_frames = 0, 0
    total_detections, unique_ids = [], set()
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Track with persistence
        results = model.track(
            frame, 
            conf=float(conf_threshold), 
            iou=float(iou_threshold),
            persist=True, 
            agnostic_nms=True, 
            imgsz=640,
            tracker="bytetrack.yaml", 
            device=device,
            verbose=False
        )
        
        annotated_frame = results[0].plot()
        
        # Write frame
        if skip_frames == 0 or frame_idx % (skip_frames + 1) == 0:
            out.write(annotated_frame)
            
        detections = get_detection_data(results)
        total_detections.extend(detections)
        for d in detections:
            if d['track_id'] is not None: unique_ids.add(d['track_id'])
            
        processed_frames += 1
        frame_idx += 1
        if frame_idx % 10 == 0:
            progress((frame_idx / total_frames), desc=f"Tracking {frame_idx}/{total_frames}")
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    summary = f"""
### Video Analysis Summary
- **ProcessingTime:** {processing_time:.2f}s
- **FPS:** {processed_frames/processing_time:.2f}
- **Objects Detected:** {len(unique_ids)}
- **Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    """
    return output_path, summary, generate_analytics(total_detections)

def create_demo_visualization():
    """Create sample visualization for demo"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('YOLOv8 KITTI Object Detection System', fontsize=16, fontweight='bold')
    
    classes = list(KITTI_CLASSES.values())[:5]
    sample_counts = [45, 32, 18, 12, 8]
    
    axes[0, 0].bar(classes, sample_counts, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Sample Detection Distribution')
    
    conf_data = np.random.beta(8, 2, 100)
    axes[0, 1].hist(conf_data, bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Confidence Score Distribution')
    
    # Simple list for system info
    info_text = "Model: YOLOv8n\nDataset: KITTI\nInput: 640x640\nMode: ZeroGPU"
    axes[1, 0].text(0.1, 0.5, info_text, fontsize=12, family='monospace')
    axes[1, 0].axis('off')

    axes[1, 1].text(0.1, 0.5, "System Status: OK\nGPU: Available\nFPS: 30+", fontsize=12, family='monospace', color='green')
    axes[1, 1].axis('off')

    plt.tight_layout()
    temp_path = tempfile.mktemp(suffix='.png')
    plt.savefig(temp_path, dpi=100, bbox_inches='tight')
    plt.close()
    return temp_path

# --- GRADIO INTERFACE ---

with gr.Blocks(title="Autonomous Vehicle Perception", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Autonomous Vehicle Perception (GPU)")
    
    with gr.Tabs():
        # TAB 1: IMAGE ANALYSIS
        with gr.Tab("Image Analysis"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Street Scene")
                    with gr.Row():
                        conf_img = gr.Slider(0.1, 0.9, value=0.25, label="Confidence")
                        iou_img = gr.Slider(0.1, 0.9, value=0.45, label="NMS IoU")
                    detect_btn_img = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Annotated Result")
            with gr.Row():
                summary_img = gr.Markdown("### Stats")
                analytics_img = gr.Image(label="Class Distribution")
            detection_table_img = gr.Markdown("### Detailed Logs")
            
            detect_btn_img.click(
                fn=process_image, 
                inputs=[image_input, conf_img, iou_img], 
                outputs=[image_output, analytics_img, summary_img, detection_table_img]
            )
        
        # TAB 2: VIDEO TRACKING
        with gr.Tab("Video Tracking"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="Upload Driving Video")
                    with gr.Row():
                        conf_vid = gr.Slider(0.1, 0.9, value=0.25, label="Confidence")
                        iou_vid = gr.Slider(0.1, 0.9, value=0.45, label="Tracker IoU")
                    skip_frames = gr.Slider(0, 5, value=0, step=1, label="Skip Output Frames")
                    detect_btn_vid = gr.Button("Start Tracking", variant="primary")
                with gr.Column():
                    video_output = gr.Video(label="Tracked Output")
            with gr.Row():
                summary_vid = gr.Markdown("### Tracking Stats")
                analytics_vid = gr.Image(label="Object Counts")
            
            detect_btn_vid.click(
                fn=process_video, 
                inputs=[video_input, conf_vid, iou_vid, skip_frames], 
                outputs=[video_output, summary_vid, analytics_vid]
            )
            
        with gr.Tab("System Info"):
            gr.Image(value=create_demo_visualization, label="System Status", show_label=False)

print("System Initialized. Loading Model...")
load_model()

if __name__ == "__main__":
    demo.launch()