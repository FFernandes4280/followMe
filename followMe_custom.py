import cv2
import numpy as np
import argparse

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
    
    # Se toda a grade está ocupada, recuar
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

def parse_model_output(outputs, conf_threshold=0.3, target_class=None):
    """
    Processa a saída do modelo ONNX de forma genérica
    
    Args:
        outputs: Saída do modelo
        conf_threshold: Threshold de confiança
        target_class: Classe alvo (None = todas as classes)
    
    Returns:
        boxes, confidences, class_ids
    """
    outputs = outputs[0]
    
    # Transpõe se necessário para ter formato (num_detections, num_values)
    if len(outputs.shape) == 3:
        # Remove dimensão extra, ex: (1, 84, 8400) -> (84, 8400)
        outputs = outputs.squeeze(0)
    
    # Se primeiro dim é menor (ex: 84 vs 8400), transpor
    if len(outputs.shape) == 2 and outputs.shape[0] < outputs.shape[1]:
        outputs = outputs.T
    
    rows = outputs.shape[0]
    num_features = outputs.shape[1]
    
    boxes, confidences, class_ids = [], [], []
    
    # Determina quantas classes o modelo tem
    # Assumindo formato: [cx, cy, w, h, class_scores...]
    num_classes = num_features - 4
    
    for i in range(rows):
        detection = outputs[i]
        
        # Coordenadas da caixa (normalmente primeiros 4 valores)
        cx, cy, w, h = detection[0:4]
        
        # Scores das classes (resto dos valores)
        scores = detection[4:] 
        
        # Encontra a classe com maior score
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Filtra por threshold e classe alvo (se especificada)
        if confidence > conf_threshold:
            if target_class is None or class_id == target_class:
                boxes.append([cx, cy, w, h])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    
    return boxes, confidences, class_ids, num_classes

def scale_boxes(boxes, cx_cy_wh_format, image_width, image_height, input_size=640):
    """
    Converte boxes de coordenadas normalizadas para pixels
    
    Args:
        boxes: Lista de boxes [cx, cy, w, h] ou [x, y, w, h]
        cx_cy_wh_format: True se boxes estão em formato centro+tamanho
        image_width, image_height: Tamanho da imagem de destino
        input_size: Tamanho de entrada do modelo
    """
    scaled_boxes = []
    
    for box in boxes:
        if cx_cy_wh_format:
            # Formato centro + largura/altura
            cx, cy, w, h = box
            x = int((cx - w/2) * image_width / input_size)
            y = int((cy - h/2) * image_height / input_size)
            width = int(w * image_width / input_size)
            height = int(h * image_height / input_size)
        else:
            # Formato x, y, width, height já em pixels
            x, y, width, height = box
        
        # Garante coordenadas dentro dos limites
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
        
        # Define cor baseada na classe
        color = (0, 255, 0)  # Verde padrão
        
        # Desenha retângulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Classe {class_id}: {confidence:.2f}"
        
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

def main():
    parser = argparse.ArgumentParser(description='Sistema de rastreamento com modelo ONNX customizado')
    parser.add_argument('--model', type=str, required=True, 
                        help='Caminho para o modelo ONNX (ex: long.onnx ou short.onnx)')
    parser.add_argument('--source', type=str, default='0',
                        help='Fonte de vídeo (0 para webcam, ou caminho para vídeo)')
    
    args = parser.parse_args()
    
    # Carrega modelo silenciosamente
    net = cv2.dnn.readNetFromONNX(args.model)
    
    # Abre fonte de vídeo
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a fonte de vídeo")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    movement_threshold = 0.15
    print_occupation_grid = False
    show_grid_overlay = True  # Always show grid with visual feedback
    paused = False
    num_classes = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            image_height, image_width = frame.shape[:2]
            
            # Prepara frame para detecção
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Cria blob e faz inferência
            blob = cv2.dnn.blobFromImage(
                detection_frame, 1/255.0, (640, 640), 
                swapRB=True, crop=False
            )
            net.setInput(blob)
            
            # Inferência
            outputs = net.forward()
            
            # Processa saída do modelo
            raw_boxes, confidences, class_ids, detected_classes = parse_model_output(
                outputs, 
                conf_threshold=0.3,
                target_class=None
            )
            
            if num_classes is None:
                num_classes = detected_classes
            
            # Converte boxes para pixels
            boxes = scale_boxes(
                raw_boxes, 
                cx_cy_wh_format=True,
                image_width=image_width,
                image_height=image_height
            )
            
            # Aplica NMS
            indices = []
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
            
            # Calcula ocupação da grade
            occupation = calculate_grid_occupation(
                boxes, indices if len(boxes) > 0 else [], 
                image_width, image_height
            )
            
            # Gera comandos de movimento
            movement_commands = generate_movement_commands(occupation, movement_threshold)
            
            # Imprime apenas comandos de movimento
            if movement_commands:
                for cmd in movement_commands:
                    print(f"CMD: {cmd}")
            
            # Desenha detecções
            display_frame = frame.copy()
            if len(indices) > 0:
                draw_detections(
                    display_frame, boxes, confidences, class_ids, 
                    indices, None
                )
            
            # Desenha grade com feedback visual de ocupação
            if show_grid_overlay:
                draw_grid(display_frame, occupation)
            
            # Adiciona informações mínimas no frame
            info_text = f"Deteccoes: {len(indices)}"
            cv2.putText(
                display_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("Rastreamento Customizado", display_frame)
        else:
            # Modo pausado - continua mostrando o último frame
            cv2.imshow("Rastreamento Customizado", display_frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            print_occupation_grid = not print_occupation_grid
        elif key == ord("g"):
            show_grid_overlay = not show_grid_overlay
        elif key == ord("p"):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
