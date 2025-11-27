import cv2
import numpy as np
import argparse
import os
import time

def calculate_grid_occupation(boxes, indices, image_width, image_height):
    """Calcula ocupação de cada quadrado da grade (3x5 para portrait, 3x3 para landscape)"""
    grid_base_width = image_width
    grid_base_height = image_height
    
    # Determina grid baseado na orientação
    if grid_base_height > grid_base_width:
        # Portrait: 3 colunas x 5 linhas
        grid_cols = 3
        grid_rows = 5
    else:
        # Landscape: 3x3
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
    """Gera comandos de movimento baseado na ocupação da grade
    
    Como a câmera do celular já é a câmera do drone, os comandos indicam
    para onde o usuário deve mover fisicamente o celular.
    """
    commands = []
    grid_rows, grid_cols = occupation.shape
    center_row = grid_rows // 2
    center_col = grid_cols // 2
    center = occupation[center_row, center_col]
    left_col = np.sum(occupation[:, 0])
    right_col = np.sum(occupation[:, grid_cols - 1])
    top_row = np.sum(occupation[0, :])
    bottom_row = np.sum(occupation[grid_rows - 1, :])
    
    # Se toda a grade está ocupada, recuar
    if np.all(occupation > 0.1):
        commands.append("⬅️ RECUAR - Afaste o celular")
        return commands
    
    high_occ = 0.6
    if center > high_occ and np.sum(occupation) - center < threshold:
        commands.append("➡️ SEGUIR_FRENTE - Aproxime o celular")
    
    if left_col > threshold * 2 and right_col < threshold:
        commands.append("↪️ VIRAR_ESQUERDA - Mova celular para esquerda")
    elif right_col > threshold * 2 and left_col < threshold:
        commands.append("↩️ VIRAR_DIREITA - Mova celular para direita")
    
    if top_row > threshold * 2 and bottom_row < threshold:
        commands.append("⬆️ INCLINAR_PARA_CIMA - Aponte celular para cima")
    elif bottom_row > threshold * 2 and top_row < threshold:
        commands.append("⬇️ INCLINAR_PARA_BAIXO - Aponte celular para baixo")
    
    if np.sum(occupation) < threshold:
        commands.append("❌ ALVO PERDIDO - Procure ao redor")
    
    return commands

def parse_model_output(outputs, conf_threshold=0.3, target_class=None):
    """Processa a saída do modelo ONNX de forma genérica"""
    outputs = outputs[0]
    
    if len(outputs.shape) == 3:
        outputs = outputs.squeeze(0)
    
    if len(outputs.shape) == 2 and outputs.shape[0] < outputs.shape[1]:
        outputs = outputs.T
    
    rows = outputs.shape[0]
    num_features = outputs.shape[1]
    
    boxes, confidences, class_ids = [], [], []
    num_classes = num_features - 4
    
    for i in range(rows):
        detection = outputs[i]
        cx, cy, w, h = detection[0:4]
        scores = detection[4:] 
        
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > conf_threshold:
            if target_class is None or class_id == target_class:
                boxes.append([cx, cy, w, h])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    
    return boxes, confidences, class_ids, num_classes

def scale_boxes(boxes, cx_cy_wh_format, image_width, image_height, input_size=640):
    """Converte boxes de coordenadas normalizadas para pixels"""
    scaled_boxes = []
    
    for box in boxes:
        if cx_cy_wh_format:
            cx, cy, w, h = box
            x = int((cx - w/2) * image_width / input_size)
            y = int((cy - h/2) * image_height / input_size)
            width = int(w * image_width / input_size)
            height = int(h * image_height / input_size)
        else:
            x, y, width, height = box
        
        x = max(0, x)
        y = max(0, y)
        width = min(width, image_width - x)
        height = min(height, image_height - y)
        
        scaled_boxes.append([x, y, width, height])
    
    return scaled_boxes

def draw_detections(frame, boxes, confidences, class_ids, indices, class_names=None):
    """Desenha as detecções no frame"""
    for i in indices:
        x, y, w, h = boxes[i]
        confidence = confidences[i]
        class_id = class_ids[i]
        
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Classe {class_id}: {confidence:.2f}"
        
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

def draw_grid(frame, occupation=None):
    """Desenha a grade no frame com feedback visual de ocupação"""
    height, width = frame.shape[:2]
    
    # Determina grid baseado na orientação
    if height > width:
        grid_cols = 3
        grid_rows = 5
    else:
        grid_cols = 3
        grid_rows = 3
    
    grid_width = width // grid_cols
    grid_height = height // grid_rows
    
    # Desenha overlay colorido se temos dados de ocupação
    if occupation is not None:
        overlay = frame.copy()
        for row in range(grid_rows):
            for col in range(grid_cols):
                occ = occupation[row, col]
                if occ > 0:
                    green_value = int(100 + (155 * (1 - occ)))
                    color = (0, green_value, 0)
                    
                    x1 = col * grid_width
                    y1 = row * grid_height
                    x2 = x1 + grid_width
                    y2 = y1 + grid_height
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    
                    text = f"{occ:.0%}"
                    font_scale = 0.8
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    text_x = x1 + (grid_width - text_width) // 2
                    text_y = y1 + (grid_height + text_height) // 2
                    
                    cv2.putText(overlay, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness + 2)
                    cv2.putText(overlay, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Linhas da grade
    for i in range(1, grid_cols):
        x = i * grid_width
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 2)
    
    for i in range(1, grid_rows):
        y = i * grid_height
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 2)

def draw_hud(frame, commands, frame_count, fps):
    """Desenha HUD com comandos de movimento"""
    height, width = frame.shape[:2]
    
    # Painel de comandos no topo
    panel_h = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Título
    cv2.putText(frame, "PHONE DRONE CAMERA", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Status
    status_text = f"Frame: {frame_count} | FPS: {fps:.1f}"
    cv2.putText(frame, status_text, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Comandos
    if commands:
        y_offset = 90
        for cmd in commands:
            cv2.putText(frame, cmd, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
    else:
        cv2.putText(frame, "✓ CENTRALIZED - Keep position", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    parser = argparse.ArgumentParser(description='Sistema de rastreamento com câmera do celular como drone')
    parser.add_argument('--model', type=str, required=True, 
                        help='Caminho para o modelo ONNX')
    parser.add_argument('--source', type=str, default='http://192.168.3.43:8080/video',
                        help='URL da câmera do celular')
    
    args = parser.parse_args()
    
    # Carrega modelo
    net = cv2.dnn.readNetFromONNX(args.model)
    
    # Abre câmera do celular
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível conectar à câmera do celular")
        print("\nVerifique se:")
        print("1. O app IP Webcam está rodando")
        print("2. Dispositivos estão na mesma rede WiFi")
        print("3. A URL está correta")
        return
    
    print("✓ Conectado à câmera do celular!")
    print("\nControles:")
    print("  q - Sair")
    print("  g - Toggle grid overlay")
    print("  p - Pausar/Continuar")
    print("\n🎯 Siga os comandos na tela para posicionar o celular\n")
    
    frame_count = 0
    movement_threshold = 0.15
    show_grid_overlay = True
    paused = False
    
    start_time = time.time()
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Erro ao capturar frame")
                break
            
            frame_count += 1
            image_height, image_width = frame.shape[:2]
            
            # Cria blob e faz inferência
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (640, 640),
                swapRB=True, crop=False
            )
            net.setInput(blob)
            outputs = net.forward()
            
            # Processa saída do modelo
            raw_boxes, confidences, class_ids, detected_classes = parse_model_output(
                outputs, conf_threshold=0.4, target_class=0  # Apenas pessoa (classe 0)
            )
            
            # Converte boxes para pixels
            boxes = scale_boxes(
                raw_boxes, cx_cy_wh_format=True,
                image_width=image_width, image_height=image_height
            )
            
            # Aplica NMS
            indices = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)
            
            # Calcula ocupação da grade
            occupation = calculate_grid_occupation(
                boxes, indices if len(boxes) > 0 else [], 
                image_width, image_height
            )
            
            # Gera comandos de movimento
            commands = generate_movement_commands(occupation, movement_threshold)
            
            # Calcula FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Desenha detecções
            if len(indices) > 0:
                draw_detections(frame, boxes, confidences, class_ids, indices, None)
            
            if show_grid_overlay:
                draw_grid(frame, occupation)
            
            draw_hud(frame, commands, frame_count, fps)
            
            # Mostra frame
            cv2.imshow("Phone Drone Camera", frame)
        else:
            cv2.imshow("Phone Drone Camera", frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            show_grid_overlay = not show_grid_overlay
        elif key == ord("p"):
            paused = not paused
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    elapsed_total = time.time() - start_time
    print(f"\n📊 Total de frames: {frame_count}")
    print(f"   FPS médio: {frame_count / elapsed_total:.1f}")

if __name__ == "__main__":
    main()
