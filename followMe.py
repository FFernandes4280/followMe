
import cv2
import numpy as np

# Carrega o modelo treinado com dataset do Red Bull
net = cv2.dnn.readNetFromONNX("sports_detection_best.onnx")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

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
print("Sistema de comandos PID por grade 3x3. Pressione 'q' para sair. Pressione 'o' para alternar impressão da grade no terminal. Pressione 'c' para alternar modo colorido/preto e branco.")

frame_count = 0
movement_threshold = 0.15  # Threshold para comandos de movimento
print_occupation_grid = False
color_mode = True  # True = colorido, False = preto e branco  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image_height, image_width = frame.shape[:2]
    
    # Aplica modo preto e branco se necessário
    if not color_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        detection_frame = frame.copy()
    blob = cv2.dnn.blobFromImage(detection_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    try:
        outputs = net.forward()
        print(f"Model output shape: {outputs.shape}")
        outputs = outputs[0]  
        print(f"First output shape: {outputs.shape}")
        
        # Transpõe se necessário para ter formato (num_detections, num_values)
        if outputs.shape[0] == 84:  # Se está no formato (84, 8400)
            outputs = outputs.T  # Transpõe para (8400, 84)
            print(f"After transpose: {outputs.shape}")
    except Exception as e:
        print(f"Error in model forward: {e}")
        continue

    rows = outputs.shape[0]
    boxes, confidences, class_ids = [], [], []

    for i in range(rows):
        detection = outputs[i]
        
        cx, cy, w, h = detection[0:4]
        confidence = detection[4]  # Apenas o score de confiança
        
        # Debug: mostra todas as detecções
        if confidence > 0.05:  # Mostra detecções com confiança > 5%
            print(f"Detecção: confidence={confidence:.3f}, cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}")
        
        if confidence > 0.1:  # Apenas verifica confiança, classe é sempre 0
            # Converte coordenadas YOLO (centro, largura, altura) para formato OpenCV (x, y, w, h)
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
            class_ids.append(0)  # Sempre classe 0 (person_sporting)

    indices = []
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

    occupation = calculate_grid_occupation(boxes, indices if len(boxes) > 0 else [], image_width, image_height)
 
    movement_commands = generate_movement_commands(occupation, movement_threshold)
   
    if movement_commands:
        for cmd in movement_commands:
            print(f"CMD: {cmd}")

    # Desenha bounding boxes nas pessoas detectadas com cores diferentes
    if color_mode:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    else:
        colors = [(255, 255, 255), (200, 200, 200), (150, 150, 150), (100, 100, 100), (50, 50, 50), (25, 25, 25)]
    
    if len(boxes) > 0 and len(indices) > 0:
        for idx, i in enumerate(indices):
            if isinstance(i, np.ndarray):
                i = i[0]
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            
            # Escolhe cor baseada no índice
            color = colors[idx % len(colors)]
            
            # Desenha retângulo colorido ao redor da pessoa
            cv2.rectangle(detection_frame, (x, y), (x + w, y + h), color, 3)
            
            # Adiciona texto com confiança e cor de fundo
            label = f"Pessoa {idx+1}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Desenha retângulo de fundo para o texto
            cv2.rectangle(detection_frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)
            
            # Adiciona texto branco
            cv2.putText(detection_frame, label, (x + 5, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Desenha grade 3x3 colorida
    grid_width = image_width // 3
    grid_height = image_height // 3
    
    # Cores para cada quadrante
    if color_mode:
        grid_colors = [
            (255, 0, 0),    # Vermelho - canto superior esquerdo
            (0, 255, 0),    # Verde - centro superior
            (0, 0, 255),    # Azul - canto superior direito
            (255, 255, 0),  # Amarelo - centro esquerdo
            (255, 0, 255),  # Magenta - centro
            (0, 255, 255),  # Ciano - centro direito
            (128, 0, 128),  # Roxo - canto inferior esquerdo
            (255, 165, 0),  # Laranja - centro inferior
            (0, 128, 0)     # Verde escuro - canto inferior direito
        ]
    else:
        grid_colors = [
            (255, 255, 255),  # Branco - canto superior esquerdo
            (200, 200, 200),  # Cinza claro - centro superior
            (150, 150, 150),  # Cinza médio - canto superior direito
            (100, 100, 100),  # Cinza escuro - centro esquerdo
            (50, 50, 50),     # Cinza muito escuro - centro
            (25, 25, 25),     # Quase preto - centro direito
            (200, 200, 200),  # Cinza claro - canto inferior esquerdo
            (150, 150, 150),  # Cinza médio - centro inferior
            (100, 100, 100)   # Cinza escuro - canto inferior direito
        ]
    
    # Desenha linhas da grade
    for i in range(1, 3):
        # Linhas verticais
        cv2.line(detection_frame, (i * grid_width, 0), (i * grid_width, image_height), (255, 255, 255), 2)
        # Linhas horizontais
        cv2.line(detection_frame, (0, i * grid_height), (image_width, i * grid_height), (255, 255, 255), 2)
    
    # Adiciona labels dos quadrantes
    for row in range(3):
        for col in range(3):
            quadrant_idx = row * 3 + col
            center_x = col * grid_width + grid_width // 2
            center_y = row * grid_height + grid_height // 2
            
            # Desenha círculo colorido no centro de cada quadrante
            cv2.circle(detection_frame, (center_x, center_y), 15, grid_colors[quadrant_idx], -1)
            
            # Adiciona número do quadrante
            cv2.putText(detection_frame, str(quadrant_idx + 1), 
                       (center_x - 8, center_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if print_occupation_grid:
        print("Grade de ocupação:")
        for row in range(3):
            row_str = ""
            for col in range(3):
                row_str += f"{occupation[row, col]:6.1%} "
            print(f"  [{row_str}]")
        print("-" * 50)

    # Adiciona indicador de modo no canto superior esquerdo
    mode_text = "MODO: COLORIDO" if color_mode else "MODO: P&B"
    mode_color = (0, 255, 0) if color_mode else (255, 255, 255)
    cv2.putText(detection_frame, mode_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    cv2.imshow("YOLOv8 - Pessoas", detection_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("o"):
        print_occupation_grid = not print_occupation_grid
    elif key == ord("c"):
        color_mode = not color_mode
        print(f"Modo alterado para: {'COLORIDO' if color_mode else 'PRETO E BRANCO'}")

cap.release()
cv2.destroyAllWindows()
