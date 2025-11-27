import cv2
import numpy as np
import argparse
import onnxruntime as ort
import os
import time

class PIDController:
    """Controlador PID para movimento suave do drone simulado"""
    def __init__(self, kp=0.5, ki=0.01, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def update(self, target, current):
        current_time = time.time()
        dt = current_time - self.last_time
        dt = max(dt, 0.001)
        
        error = target - current
        
        abs_error = abs(error)
        if abs_error > 100:
            adaptive_kp = self.kp * 2.5
        elif abs_error > 50:
            adaptive_kp = self.kp * 1.8
        else:
            adaptive_kp = self.kp
        
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)
        
        derivative = (error - self.previous_error) / dt
        output = adaptive_kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

class SimulatedDrone:
    def __init__(self, image_width, image_height, view_width=320, view_height=240):
        self.image_width = image_width
        self.image_height = image_height
        self.x = image_width // 2
        self.y = image_height // 2
        self.view_width = view_width
        self.view_height = view_height
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.target_x = self.x
        self.target_y = self.y
        
        self.pid_x = PIDController(kp=1.2, ki=0.03, kd=0.2)
        self.pid_y = PIDController(kp=1.2, ki=0.03, kd=0.2)
        self.pid_zoom = PIDController(kp=0.8, ki=0.02, kd=0.15)
        
        self.max_velocity = 30.0
        self.max_zoom_velocity = 0.1
    
    def set_target(self, target_x, target_y):
        self.target_x = np.clip(target_x, self.view_width // 2, 
                                self.image_width - self.view_width // 2)
        self.target_y = np.clip(target_y, self.view_height // 2, 
                                self.image_height - self.view_height // 2)
    
    def set_target_zoom(self, zoom):
        self.target_zoom = np.clip(zoom, 0.5, 3.0)
    
    def update(self):
        velocity_x = self.pid_x.update(self.target_x, self.x)
        velocity_y = self.pid_y.update(self.target_y, self.y)
        
        velocity_x = np.clip(velocity_x, -self.max_velocity, self.max_velocity)
        velocity_y = np.clip(velocity_y, -self.max_velocity, self.max_velocity)
        
        self.x += velocity_x
        self.y += velocity_y
        
        self.x = np.clip(self.x, self.view_width // 2, 
                        self.image_width - self.view_width // 2)
        self.y = np.clip(self.y, self.view_height // 2, 
                        self.image_height - self.view_height // 2)
        
        zoom_velocity = self.pid_zoom.update(self.target_zoom, self.zoom)
        zoom_velocity = np.clip(zoom_velocity, -self.max_zoom_velocity, self.max_zoom_velocity)
        
        self.zoom += zoom_velocity
        self.zoom = np.clip(self.zoom, 0.5, 3.0)
    
    def get_view_rect(self):
        effective_width = int(self.view_width / self.zoom)
        effective_height = int(self.view_height / self.zoom)
        
        x1 = int(self.x - effective_width // 2)
        y1 = int(self.y - effective_height // 2)
        x2 = x1 + effective_width
        y2 = y1 + effective_height
        
        x1 = max(0, min(x1, self.image_width - effective_width))
        y1 = max(0, min(y1, self.image_height - effective_height))
        x2 = x1 + effective_width
        y2 = y1 + effective_height
        
        return (x1, y1, x2, y2)
    
    def get_view(self, frame):
        x1, y1, x2, y2 = self.get_view_rect()
        view = frame[y1:y2, x1:x2].copy()
        view_resized = cv2.resize(view, (self.view_width, self.view_height))
        return view_resized

def calculate_grid_occupation(boxes, indices, image_width, image_height, drone_view_rect=None):
    if drone_view_rect is not None:
        x1, y1, x2, y2 = drone_view_rect
        grid_base_width = x2 - x1
        grid_base_height = y2 - y1
        grid_offset_x = x1
        grid_offset_y = y1
    else:
        grid_base_width = image_width
        grid_base_height = image_height
        grid_offset_x = 0
        grid_offset_y = 0
    
    if grid_base_height > grid_base_width:
        grid_cols = 3
        grid_rows = 5
    else:
        grid_cols = 3
        grid_rows = 3
    
    grid_width = grid_base_width // grid_cols
    grid_height = grid_base_height // grid_rows
    occupation = np.zeros((grid_rows, grid_cols))
    
    if len(boxes) == 0 or len(indices) == 0:
        return occupation
    
    if isinstance(indices, np.ndarray) and indices.ndim == 2:
        indices = indices.flatten()
    
    for i in indices:
        x, y, w, h = boxes[i]
        for row in range(grid_rows):
            for col in range(grid_cols):
                grid_x1 = grid_offset_x + col * grid_width
                grid_y1 = grid_offset_y + row * grid_height
                grid_x2 = grid_x1 + grid_width
                grid_y2 = grid_y1 + grid_height
                
                box_x1, box_y1 = x, y
                box_x2, box_y2 = x + w, y + h
                
                intersect_x1 = max(grid_x1, box_x1)
                intersect_y1 = max(grid_y1, box_y1)
                intersect_x2 = min(grid_x2, box_x2)
                intersect_y2 = min(grid_y2, box_y2)
                
                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                    grid_area = grid_width * grid_height
                    occupation[row, col] += intersect_area / grid_area
    
    return occupation

def generate_movement_commands(occupation, threshold=0.15):
    rows, cols = occupation.shape
    center_row, center_col = rows // 2, cols // 2
    
    total_occupation = np.sum(occupation)
    if total_occupation < 0.1:
        return ["Alvo perdido"], {'move_x': 0, 'move_y': 0, 'zoom_delta': 0}
    
    center_occupation = occupation[center_row, center_col]
    
    commands = []
    drone_targets = {'move_x': 0, 'move_y': 0, 'zoom_delta': 0}
    
    top_occupation = np.sum(occupation[0, :])
    bottom_occupation = np.sum(occupation[-1, :])
    left_occupation = np.sum(occupation[:, 0])
    right_occupation = np.sum(occupation[:, -1])
    
    move_step = 40
    
    if top_occupation > threshold:
        commands.append("Subir")
        drone_targets['move_y'] = -move_step
    elif bottom_occupation > threshold:
        commands.append("Descer")
        drone_targets['move_y'] = move_step
    
    if left_occupation > threshold:
        commands.append("Esquerda")
        drone_targets['move_x'] = -move_step
    elif right_occupation > threshold:
        commands.append("Direita")
        drone_targets['move_x'] = move_step
    
    zoom_in_threshold = 0.4
    zoom_out_threshold = 0.15
    
    if center_occupation > zoom_in_threshold:
        commands.append("Afastar (Zoom Out)")
        drone_targets['zoom_delta'] = -0.05
    elif center_occupation < zoom_out_threshold and total_occupation > 0.2:
        commands.append("Aproximar (Zoom In)")
        drone_targets['zoom_delta'] = 0.03
    
    if not commands:
        commands.append("Centro OK")
    
    return commands, drone_targets

def parse_onnxruntime_output(outputs, conf_threshold=0.3):
    """
    Parse ONNX Runtime output from YOLOv8 model
    Output shape: [1, num_classes+4, num_predictions] or [1, num_predictions, num_classes+4]
    """
    output = outputs[0]
    
    # Handle different output shapes
    if output.shape[1] > output.shape[2]:
        # Shape is [1, num_predictions, num_classes+4] - transpose needed
        output = output.transpose(0, 2, 1)
    
    # Now shape is [1, num_classes+4, num_predictions]
    num_classes = output.shape[1] - 4
    
    boxes = []
    confidences = []
    class_ids = []
    
    # Extract data
    for i in range(output.shape[2]):
        scores = output[0, 4:, i]  # Class scores
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            # Get box coordinates (cx, cy, w, h) - normalized
            cx, cy, w, h = output[0, :4, i]
            boxes.append([cx, cy, w, h])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
    
    return boxes, confidences, class_ids

def scale_boxes(boxes, image_width, image_height, input_size=640):
    scaled_boxes = []
    for box in boxes:
        cx, cy, w, h = box
        # Convert from normalized to pixel coordinates
        x = int((cx - w/2) * image_width / input_size)
        y = int((cy - h/2) * image_height / input_size)
        width = int(w * image_width / input_size)
        height = int(h * image_height / input_size)
        scaled_boxes.append([x, y, width, height])
    return scaled_boxes

def main():
    parser = argparse.ArgumentParser(description='FollowMe with ONNX Runtime')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--source', type=str, default='0', help='Video source')
    
    args = parser.parse_args()
    
    # Load model with ONNX Runtime
    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_size = input_shape[2] if len(input_shape) > 2 else 640
    print(f"Model loaded successfully. Input name: {input_name}, Input size: {input_size}x{input_size}")
    
    # Open video
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    movement_threshold = 0.15
    show_grid_overlay = True
    show_drone_view = True
    paused = False
    
    drone = None
    last_valid_commands = []
    last_drone_targets = {'move_x': 0, 'move_y': 0, 'zoom_delta': 0}
    frames_without_target = 0
    
    print("Starting video processing...")
    print("Controls: SPACE=pause, G=toggle grid, V=toggle drone view, Q=quit")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            image_height, image_width = frame.shape[:2]
            
            # Prepare input (detect model input size from session)
            input_shape = session.get_inputs()[0].shape
            input_size = input_shape[2] if len(input_shape) > 2 else 640
            
            input_frame = cv2.resize(frame, (input_size, input_size))
            input_frame = input_frame.transpose(2, 0, 1)  # HWC to CHW
            input_frame = input_frame.astype(np.float32) / 255.0
            input_frame = np.expand_dims(input_frame, axis=0)
            
            # Run inference
            outputs = session.run(None, {input_name: input_frame})
            
            # Parse output
            raw_boxes, confidences, class_ids = parse_onnxruntime_output(outputs, conf_threshold=0.3)
            
            # Scale boxes (use detected input size)
            boxes = scale_boxes(raw_boxes, image_width, image_height, input_size=input_size)
            
            # Apply NMS
            indices = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            
            # Initialize drone
            if drone is None:
                drone = SimulatedDrone(image_width, image_height,
                                      view_width=int(image_width * 0.5),
                                      view_height=int(image_height * 0.5))
                print(f"Drone initialized at ({drone.x}, {drone.y})")
            
            # Get drone view
            drone_view_rect = drone.get_view_rect()
            
            # Calculate grid occupation
            occupation = calculate_grid_occupation(boxes, indices if len(boxes) > 0 else [],
                                                  image_width, image_height,
                                                  drone_view_rect=drone_view_rect)
            
            # Generate commands
            movement_commands, drone_targets = generate_movement_commands(occupation, movement_threshold)
            
            # Target tracking logic
            target_lost = "Alvo perdido" in movement_commands
            
            if target_lost:
                frames_without_target += 1
                if last_valid_commands and frames_without_target <= 30:
                    movement_commands = last_valid_commands.copy()
                    drone_targets = last_drone_targets.copy()
                else:
                    movement_commands = ["PROCURANDO_ALVO"]
                    drone_targets = {'move_x': 0, 'move_y': 0, 'zoom_delta': -0.05}
            else:
                if movement_commands:
                    last_valid_commands = movement_commands.copy()
                    last_drone_targets = drone_targets.copy()
                frames_without_target = 0
            
            # Update drone
            if drone_targets['move_x'] != 0 or drone_targets['move_y'] != 0:
                target_x = drone.x + drone_targets['move_x']
                target_y = drone.y + drone_targets['move_y']
                drone.set_target(target_x, target_y)
            
            if drone_targets['zoom_delta'] != 0:
                new_zoom = drone.target_zoom + drone_targets['zoom_delta']
                drone.set_target_zoom(new_zoom)
            
            drone.update()
            
            # Draw visualizations
            display_frame = frame.copy()
            
            # Draw detections
            if len(boxes) > 0 and len(indices) > 0:
                if isinstance(indices, np.ndarray) and indices.ndim == 2:
                    indices = indices.flatten()
                for i in indices:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    conf_text = f"{confidences[i]:.2f}"
                    cv2.putText(display_frame, conf_text, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw drone view
            x1, y1, x2, y2 = drone_view_rect
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            # Draw grid
            if show_grid_overlay:
                rows, cols = occupation.shape
                grid_w = (x2 - x1) // cols
                grid_h = (y2 - y1) // rows
                
                for row in range(rows):
                    for col in range(cols):
                        gx1 = x1 + col * grid_w
                        gy1 = y1 + row * grid_h
                        gx2 = gx1 + grid_w
                        gy2 = gy1 + grid_h
                        
                        occ = occupation[row, col]
                        color = (0, int(255 * min(occ, 1.0)), 0)
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), color, -1)
                        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                        cv2.rectangle(display_frame, (gx1, gy1), (gx2, gy2), (255, 255, 255), 1)
            
            # Draw info
            info_y = 30
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(display_frame, f"Detections: {len(indices) if len(boxes) > 0 else 0}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            
            for cmd in movement_commands:
                cv2.putText(display_frame, cmd, (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                info_y += 25
            
            cv2.imshow('FollowMe - ONNX Runtime', display_frame)
            
            if show_drone_view:
                drone_view = drone.get_view(frame)
                cv2.imshow('Drone View', drone_view)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('g'):
            show_grid_overlay = not show_grid_overlay
        elif key == ord('v'):
            show_drone_view = not show_drone_view
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")

if __name__ == '__main__':
    main()
