
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

cap = cv2.VideoCapture(0)

def calculate_grid_occupation(boxes, indices, image_width, image_height):
    # Calcula ocupação de cada quadrado da grade 3x3
    grid_width = image_width // 3
    grid_height = image_height // 3
    occupation = np.zeros((3, 3))
    if len(boxes) == 0 or len(indices) == 0:
        return occupation
    if isinstance(indices, np.ndarray) and indices.ndim == 2:
        indices = indices.flatten()
    for i in indices:
        x, y, w, h = boxes[i]
        for row in range(3):
            for col in range(3):
                grid_x1 = col * grid_width
                grid_y1 = row * grid_height
                grid_x2 = grid_x1 + grid_width
                grid_y2 = grid_y1 + grid_height
                
                # Coordenadas da bounding box
                box_x1, box_y1 = x, y
                box_x2, box_y2 = x + w, y + h
                
                # Calcula interseção
                intersect_x1 = max(grid_x1, box_x1)
                intersect_y1 = max(grid_y1, box_y1)
                intersect_x2 = min(grid_x2, box_x2)
                intersect_y2 = min(grid_y2, box_y2)
                
                # Se há interseção
                if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                    box_area = w * h
                    
                    # Porcentagem da bounding box que está no quadrado
                    occupation_percentage = intersect_area / box_area if box_area > 0 else 0
                    
                    # Atualiza ocupação (pega o máximo se houver múltiplas detecções)
                    occupation[row, col] = max(occupation[row, col], occupation_percentage)
    return occupation

def generate_movement_commands(occupation, threshold=0.1):
    # Gera comandos de movimento baseado na ocupação da grade
    commands = []
    center = occupation[1, 1]
    left_col = np.sum(occupation[:, 0])
    right_col = np.sum(occupation[:, 2])
    top_row = np.sum(occupation[0, :])
    bottom_row = np.sum(occupation[2, :])
    if np.all(occupation > 0.1):
        commands.append("RECUAR")
        return commands
    high_occ = 0.6
    if center > high_occ and np.sum(occupation) - center < threshold:
        commands.append("SEGUIR_FRENTE")
    if left_col > threshold * 2 and right_col < threshold:
        commands.append("VIRAR_ESQUERDA")
    elif right_col > threshold * 2 and left_col < threshold:
        commands.append("VIRAR_DIREITA")
    if top_row > threshold * 2 and bottom_row < threshold:
        commands.append("INCLINAR_PARA_CIMA")
    elif bottom_row > threshold * 2 and top_row < threshold:
        commands.append("INCLINAR_PARA_BAIXO")
    if np.sum(occupation) < threshold:
        commands.append("Alvo perdido")
    return commands

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Sistema de comandos PID por grade 3x3. Pressione 'q' para sair. Pressione 'o' para alternar impressão da grade no terminal.")

frame_count = 0
movement_threshold = 0.15  # Threshold para comandos de movimento
print_occupation_grid = False  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image_height, image_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blob = cv2.dnn.blobFromImage(detection_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    outputs = net.forward()
    outputs = outputs[0]  
    
    # Transpõe se necessário para ter formato (num_detections, num_values)
    if outputs.shape[0] == 84:  # Se está no formato (84, 8400)
        outputs = outputs.T  # Transpõe para (8400, 84)

    rows = outputs.shape[0]
    boxes, confidences, class_ids = [], [], []

    for i in range(rows):
        detection = outputs[i]
        
        cx, cy, w, h = detection[0:4]
        scores = detection[4:] 
        
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if class_id == 0 and confidence > 0.3:
            x = int((cx - w/2) * image_width / 640)
            y = int((cy - h/2) * image_height / 640)
            width = int(w * image_width / 640)
            height = int(h * image_height / 640)
            
            x = max(0, x)
            y = max(0, y)
            width = min(width, image_width - x)
            height = min(height, image_height - y)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indices = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    occupation = calculate_grid_occupation(boxes, indices if len(boxes) > 0 else [], image_width, image_height)
 
    movement_commands = generate_movement_commands(occupation, movement_threshold)
   
    if movement_commands:
        for cmd in movement_commands:
            print(f"CMD: {cmd}")

    if print_occupation_grid:
        print("Grade de ocupação:")
        for row in range(3):
            row_str = ""
            for col in range(3):
                row_str += f"{occupation[row, col]:6.1%} "
            print(f"  [{row_str}]")
        print("-" * 50)

    cv2.imshow("YOLOv8 - Pessoas", detection_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("o"):
        print_occupation_grid = not print_occupation_grid
        print(f"Impressão da grade de ocupação: {'ON' if print_occupation_grid else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
