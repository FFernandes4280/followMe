
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

cap = cv2.VideoCapture("videos/Falls_Wont_Stop_Him.mp4")

def calculate_grid_occupation(boxes, indices, image_width, image_height):
    """Calcula ocupação de cada quadrado da grade (3x5 para portrait, 3x3 para landscape)"""
    # Determina grid baseado na orientação
    if image_height > image_width:
        # Portrait: 3 colunas x 5 linhas
        grid_cols = 3
        grid_rows = 5
    else:
        # Landscape: 3x3
        grid_cols = 3
        grid_rows = 3
    
    grid_width = image_width // grid_cols
    grid_height = image_height // grid_rows
    occupation = np.zeros((grid_rows, grid_cols))
    if len(boxes) == 0 or len(indices) == 0:
        return occupation
    if isinstance(indices, np.ndarray) and indices.ndim == 2:
        indices = indices.flatten()
    for i in indices:
        x, y, w, h = boxes[i]
        for row in range(grid_rows):
            for col in range(grid_cols):
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
    """Gera comandos de movimento baseado na ocupação da grade"""
    commands = []
    grid_rows, grid_cols = occupation.shape
    
    # Centro adapta-se ao tamanho da grade
    center_row = grid_rows // 2
    center_col = grid_cols // 2
    center = occupation[center_row, center_col]
    
    left_col = np.sum(occupation[:, 0])
    right_col = np.sum(occupation[:, grid_cols - 1])
    top_row = np.sum(occupation[0, :])
    bottom_row = np.sum(occupation[grid_rows - 1, :])
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

def draw_grid(frame, occupation=None):
    """Desenha a grade 3x3 no frame com feedback visual de ocupação"""
    height, width = frame.shape[:2]
    
    # Para vídeos portrait (altura > largura), ajusta para divisões mais proporcionais
    # Usa divisão em 3 colunas e 5 linhas para melhor proporção
    if height > width:
        grid_cols = 3
        grid_rows = 5
        grid_width = width // grid_cols
        grid_height = height // grid_rows
    else:
        # Vídeos landscape mantêm 3x3
        grid_cols = 3
        grid_rows = 3
        grid_width = width // grid_cols
        grid_height = height // grid_rows
    
    # Se temos dados de ocupação, desenha overlay colorido
    if occupation is not None:
        overlay = frame.copy()
        for row in range(grid_rows):
            for col in range(grid_cols):
                occ = occupation[row, col]
                if occ > 0:
                    # Calcula intensidade do verde (mais escuro = maior ocupação)
                    # Ocupação 0% = sem cor, 100% = verde escuro (0, 100, 0)
                    intensity = int(255 * (1 - occ))  # Inverte: maior ocupação = mais escuro
                    green_value = int(100 + (155 * (1 - occ)))  # Verde de 100 a 255
                    color = (0, green_value, 0)  # BGR format
                    
                    # Coordenadas do quadrado
                    x1 = col * grid_width
                    y1 = row * grid_height
                    x2 = x1 + grid_width
                    y2 = y1 + grid_height
                    
                    # Desenha retângulo preenchido com transparência
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    
                    # Adiciona texto com a porcentagem
                    text = f"{occ:.0%}"
                    font_scale = 0.8
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    text_x = x1 + (grid_width - text_width) // 2
                    text_y = y1 + (grid_height + text_height) // 2
                    
                    # Texto com contorno para melhor visibilidade
                    cv2.putText(overlay, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2)
                    cv2.putText(overlay, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        # Aplica overlay com transparência
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Linhas da grade (sempre visíveis)
    # Linhas verticais
    for i in range(1, grid_cols):
        x = i * grid_width
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 2)
    
    # Linhas horizontais
    for i in range(1, grid_rows):
        y = i * grid_height
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 2)

def draw_detections(frame, boxes, confidences, indices):
    """Desenha as detecções no frame"""
    for i in indices:
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        
        color = (0, 255, 0)  # Verde
        
        # Desenha retângulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Label
        label = f"Pessoa: {confidence:.2f}"
        
        # Desenha label com fundo
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (x, y - label_height - baseline),
            (x + label_width, y),
            color,
            -1
        )
        cv2.putText(
            frame, label, (x, y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
movement_threshold = 0.15
print_occupation_grid = False
show_grid_overlay = True  

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
   
    # Imprime apenas comandos de movimento
    if movement_commands:
        for cmd in movement_commands:
            print(f"CMD: {cmd}")

    # Desenha detecções e grade no frame original
    display_frame = frame.copy()
    if len(indices) > 0:
        draw_detections(display_frame, boxes, confidences, indices)
    
    # Desenha grade com feedback visual de ocupação
    if show_grid_overlay:
        draw_grid(display_frame, occupation)
    
    # Adiciona informações mínimas no frame
    info_text = f"Deteccoes: {len(indices)}"
    cv2.putText(
        display_frame, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    cv2.imshow("YOLOv8 - Pessoas (COCO)", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("o"):
        print_occupation_grid = not print_occupation_grid
    elif key == ord("g"):
        show_grid_overlay = not show_grid_overlay

cap.release()
cv2.destroyAllWindows()
