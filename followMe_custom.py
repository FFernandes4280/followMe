import cv2
import numpy as np
import argparse
import os
import time

class PIDController:
    """Controlador PID para movimento suave do drone simulado"""
    def __init__(self, kp=0.5, ki=0.01, kd=0.1):
        self.kp = kp  # Proporcional
        self.ki = ki  # Integral
        self.kd = kd  # Derivativo
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def update(self, target, current):
        """Calcula a saída do PID com ganho adaptativo baseado no erro"""
        current_time = time.time()
        dt = current_time - self.last_time
        dt = max(dt, 0.001)  # Evita divisão por zero
        
        error = target - current
        
        # Ganho adaptativo: quanto maior o erro, maior o Kp
        # Isso faz o drone se mover mais rápido quando está longe do alvo
        abs_error = abs(error)
        if abs_error > 100:  # Erro grande
            adaptive_kp = self.kp * 2.5  # 2.5x mais rápido
        elif abs_error > 50:  # Erro médio
            adaptive_kp = self.kp * 1.8  # 1.8x mais rápido
        else:  # Erro pequeno
            adaptive_kp = self.kp  # Velocidade normal
        
        self.integral += error * dt
        
        # Anti-windup: limita o termo integral
        self.integral = np.clip(self.integral, -100, 100)
        
        derivative = (error - self.previous_error) / dt
        
        # Usa o Kp adaptativo
        output = adaptive_kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reseta o controlador PID"""
        self.previous_error = 0
        self.integral = 0
        self.last_time = time.time()

class SimulatedDrone:
    """Representa o drone simulado com posição, zoom e controladores PID"""
    def __init__(self, image_width, image_height, view_width=320, view_height=240):
        self.image_width = image_width
        self.image_height = image_height
        
        # Posição central inicial
        self.x = image_width // 2
        self.y = image_height // 2
        
        # Dimensões da visão do drone
        self.view_width = view_width
        self.view_height = view_height
        
        # Zoom (1.0 = normal, >1 = mais próximo)
        self.zoom = 1.0
        self.target_zoom = 1.0
        
        # Posições alvo
        self.target_x = self.x
        self.target_y = self.y
        
        # Controladores PID para X, Y e Zoom (ganhos aumentados)
        self.pid_x = PIDController(kp=1.2, ki=0.03, kd=0.2)
        self.pid_y = PIDController(kp=1.2, ki=0.03, kd=0.2)
        self.pid_zoom = PIDController(kp=0.8, ki=0.02, kd=0.15)
        
        # Velocidades máximas (pixels/frame) - aumentadas
        self.max_velocity = 30.0  # Era 15.0
        self.max_zoom_velocity = 0.1  # Era 0.05
    
    def set_target(self, target_x, target_y):
        """Define a posição alvo para o drone"""
        self.target_x = np.clip(target_x, self.view_width // 2, 
                                self.image_width - self.view_width // 2)
        self.target_y = np.clip(target_y, self.view_height // 2, 
                                self.image_height - self.view_height // 2)
    
    def set_target_zoom(self, zoom):
        """Define o zoom alvo"""
        self.target_zoom = np.clip(zoom, 0.5, 3.0)
    
    def update(self):
        """Atualiza a posição do drone usando controladores PID"""
        # Calcula as saídas do PID
        velocity_x = self.pid_x.update(self.target_x, self.x)
        velocity_y = self.pid_y.update(self.target_y, self.y)
        velocity_zoom = self.pid_zoom.update(self.target_zoom, self.zoom)
        
        # Limita velocidades
        velocity_x = np.clip(velocity_x, -self.max_velocity, self.max_velocity)
        velocity_y = np.clip(velocity_y, -self.max_velocity, self.max_velocity)
        velocity_zoom = np.clip(velocity_zoom, -self.max_zoom_velocity, self.max_zoom_velocity)
        
        # Atualiza posições
        self.x += velocity_x
        self.y += velocity_y
        self.zoom += velocity_zoom
        
        # Garante que o drone não saia dos limites
        self.x = np.clip(self.x, self.view_width // 2, 
                        self.image_width - self.view_width // 2)
        self.y = np.clip(self.y, self.view_height // 2, 
                        self.image_height - self.view_height // 2)
        self.zoom = np.clip(self.zoom, 0.5, 3.0)
        
        return velocity_x, velocity_y, velocity_zoom
    
    def get_view_rect(self):
        """Retorna o retângulo da visão do drone (x1, y1, x2, y2)"""
        half_w = int(self.view_width / (2 * self.zoom))
        half_h = int(self.view_height / (2 * self.zoom))
        
        x1 = int(self.x - half_w)
        y1 = int(self.y - half_h)
        x2 = int(self.x + half_w)
        y2 = int(self.y + half_h)
        
        # Garante que está dentro dos limites
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.image_width, x2)
        y2 = min(self.image_height, y2)
        
        return x1, y1, x2, y2

def calculate_grid_occupation(boxes, indices, image_width, image_height, drone_view_rect=None):
    """Calcula ocupação de cada quadrado da grade (3x5 para portrait, 3x3 para landscape)
    
    Se drone_view_rect for fornecido, o grid é calculado apenas dentro da visão do drone.
    """
    # Se há uma visão do drone, usa as dimensões dela
    if drone_view_rect is not None:
        x1, y1, x2, y2 = drone_view_rect
        grid_base_width = x2 - x1
        grid_base_height = y2 - y1
        grid_offset_x = x1
        grid_offset_y = y1
    else:
        # Usa a imagem inteira
        grid_base_width = image_width
        grid_base_height = image_height
        grid_offset_x = 0
        grid_offset_y = 0
    
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
                # Calcula posição do grid considerando offset (visão do drone)
                grid_x1 = grid_offset_x + col * grid_width
                grid_y1 = grid_offset_y + row * grid_height
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
    
    # Targets para o drone simulado
    drone_targets = {'move_x': 0, 'move_y': 0, 'zoom_delta': 0}
    
    # Se toda a grade está ocupada, recuar
    if np.all(occupation > 0.1):
        commands.append("RECUAR")
        drone_targets['zoom_delta'] = -0.2
        return commands, drone_targets
    high_occ = 0.6
    if center > high_occ and np.sum(occupation) - center < threshold:
        commands.append("SEGUIR_FRENTE")
        drone_targets['zoom_delta'] = 0.1
    if left_col > threshold * 2 and right_col < threshold:
        commands.append("VIRAR_ESQUERDA")
        # Movimento proporcional à intensidade da ocupação
        intensity = min(left_col / (threshold * 2), 2.0)
        drone_targets['move_x'] = -80 * intensity  # Até -160 pixels
    elif right_col > threshold * 2 and left_col < threshold:
        commands.append("VIRAR_DIREITA")
        intensity = min(right_col / (threshold * 2), 2.0)
        drone_targets['move_x'] = 80 * intensity  # Até +160 pixels
    if top_row > threshold * 2 and bottom_row < threshold:
        commands.append("INCLINAR_PARA_CIMA")
        intensity = min(top_row / (threshold * 2), 2.0)
        drone_targets['move_y'] = -80 * intensity  # Até -160 pixels
    elif bottom_row > threshold * 2 and top_row < threshold:
        commands.append("INCLINAR_PARA_BAIXO")
        intensity = min(bottom_row / (threshold * 2), 2.0)
        drone_targets['move_y'] = 80 * intensity  # Até +160 pixels
    if np.sum(occupation) < threshold:
        commands.append("Alvo perdido")
    return commands, drone_targets

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

def draw_grid(frame, occupation=None, drone_view_rect=None):
    """Desenha a grade 3x3 no frame com feedback visual de ocupação
    
    Se drone_view_rect for fornecido, desenha apenas dentro da visão do drone.
    """
    # Se há uma visão do drone, usa as dimensões dela
    if drone_view_rect is not None:
        x1, y1, x2, y2 = drone_view_rect
        grid_base_width = x2 - x1
        grid_base_height = y2 - y1
        grid_offset_x = x1
        grid_offset_y = y1
    else:
        # Usa a imagem inteira
        height, width = frame.shape[:2]
        grid_base_width = width
        grid_base_height = height
        grid_offset_x = 0
        grid_offset_y = 0
    
    # Para vídeos portrait (altura > largura), ajusta para divisões mais proporcionais
    # Usa divisão em 3 colunas e 5 linhas para melhor proporção
    if grid_base_height > grid_base_width:
        grid_cols = 3
        grid_rows = 5
        grid_width = grid_base_width // grid_cols
        grid_height = grid_base_height // grid_rows
    else:
        # Vídeos landscape mantêm 3x3
        grid_cols = 3
        grid_rows = 3
        grid_width = grid_base_width // grid_cols
        grid_height = grid_base_height // grid_rows
    
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
                    
                    # Coordenadas do quadrado (com offset para visão do drone)
                    x1 = grid_offset_x + col * grid_width
                    y1 = grid_offset_y + row * grid_height
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
    
    # Linhas da grade (sempre visíveis) - com offset para visão do drone
    # Linhas verticais
    for i in range(1, grid_cols):
        x = grid_offset_x + i * grid_width
        y1_line = grid_offset_y
        y2_line = grid_offset_y + grid_base_height
        cv2.line(frame, (x, y1_line), (x, y2_line), (255, 255, 255), 2)
    
    # Linhas horizontais
    for i in range(1, grid_rows):
        y = grid_offset_y + i * grid_height
        x1_line = grid_offset_x
        x2_line = grid_offset_x + grid_base_width
        cv2.line(frame, (x1_line, y), (x2_line, y), (255, 255, 255), 2)

def draw_drone_view_rectangle(frame, drone):
    """Desenha o retângulo que representa a visão do drone"""
    x1, y1, x2, y2 = drone.get_view_rect()
    
    # Retângulo externo (azul ciano)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
    
    # Linha de crosshair no centro
    center_x = int(drone.x)
    center_y = int(drone.y)
    cross_size = 10
    cv2.line(frame, (center_x - cross_size, center_y), 
             (center_x + cross_size, center_y), (0, 255, 255), 2)
    cv2.line(frame, (center_x, center_y - cross_size), 
             (center_x, center_y + cross_size), (0, 255, 255), 2)
    
    # Label "DRONE VIEW"
    label = f"DRONE VIEW (Zoom: {drone.zoom:.2f}x)"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def draw_hud(frame, drone, commands, velocity_x, velocity_y, frame_count):
    """Desenha HUD (Head-Up Display) com informações do drone"""
    height, width = frame.shape[:2]
    
    # Painel semi-transparente no canto superior direito
    panel_w = 280
    panel_h = 180
    panel_x = width - panel_w - 10
    panel_y = 10
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + panel_w, panel_y + panel_h), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Borda do painel
    cv2.rectangle(frame, (panel_x, panel_y), 
                  (panel_x + panel_w, panel_y + panel_h), 
                  (0, 255, 255), 2)
    
    # Informações do drone
    y_offset = panel_y + 25
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 255)
    thickness = 1
    
    infos = [
        f"DRONE STATUS",
        f"Pos: ({int(drone.x)}, {int(drone.y)})",
        f"Zoom: {drone.zoom:.2f}x",
        f"Vel: ({velocity_x:.1f}, {velocity_y:.1f})",
        f"Frame: {frame_count}",
        f"Cmds: {', '.join(commands) if commands else 'IDLE'}"
    ]
    
    for i, info in enumerate(infos):
        if i == 0:  # Título
            cv2.putText(frame, info, (panel_x + 10, y_offset),
                       font, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, info, (panel_x + 10, y_offset + i * line_height),
                       font, font_scale, color, thickness)

def extract_drone_view(frame, drone):
    """Extrai e redimensiona a visão do drone do frame"""
    x1, y1, x2, y2 = drone.get_view_rect()
    
    # Extrai a região
    drone_view = frame[y1:y2, x1:x2].copy()
    
    # Redimensiona para tamanho fixo
    if drone_view.shape[0] > 0 and drone_view.shape[1] > 0:
        drone_view = cv2.resize(drone_view, (drone.view_width, drone.view_height))
        
        # Adiciona borda
        cv2.rectangle(drone_view, (0, 0), 
                     (drone.view_width - 1, drone.view_height - 1), 
                     (255, 255, 0), 2)
        
        # Adiciona crosshair
        center_x = drone.view_width // 2
        center_y = drone.view_height // 2
        cv2.line(drone_view, (center_x - 20, center_y), 
                (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(drone_view, (center_x, center_y - 20), 
                (center_x, center_y + 20), (0, 255, 255), 2)
        
        # Label
        cv2.putText(drone_view, "DRONE CAMERA", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return drone_view

def main():
    parser = argparse.ArgumentParser(description='Sistema de rastreamento com modelo ONNX customizado')
    parser.add_argument('--model', type=str, required=True, 
                        help='Caminho para o modelo ONNX (ex: long.onnx ou short.onnx)')
    parser.add_argument('--source', type=str, default='0',
                        help='Fonte de vídeo (0 para webcam, ou caminho para vídeo)')
    
    args = parser.parse_args()
    
    net = cv2.dnn.readNetFromONNX(args.model)
    
    # Abre fonte de vídeo
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a fonte de vídeo")
        return
    
    # Aumenta a resolução para visualização melhor
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    movement_threshold = 0.15
    show_grid_overlay = True
    show_drone_view = True
    paused = False
    
    # Inicializa o drone simulado (será configurado após primeiro frame)
    # Visão maior para encontrar alvo mais rápido
    drone = None
    velocity_x, velocity_y = 0, 0
    
    # Memória dos últimos comandos válidos (quando tinha alvo)
    last_valid_commands = []
    last_drone_targets = {'move_x': 0, 'move_y': 0, 'zoom_delta': 0}
    frames_without_target = 0
    
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
            
            # Inicializa drone no primeiro frame com visão maior
            if drone is None:
                # Visão maior (50% da tela) para encontrar alvos mais rápido
                drone = SimulatedDrone(image_width, image_height, 
                                      view_width=int(image_width * 0.5), 
                                      view_height=int(image_height * 0.5))
                print(f"[INFO] Drone simulado inicializado em ({drone.x}, {drone.y})")
                print(f"[INFO] Visão do drone: {drone.view_width}x{drone.view_height}")
            
            # Obtém o retângulo da visão do drone
            drone_view_rect = drone.get_view_rect()
            
            # Calcula ocupação da grade APENAS dentro da visão do drone
            occupation = calculate_grid_occupation(
                boxes, indices if len(boxes) > 0 else [], 
                image_width, image_height,
                drone_view_rect=drone_view_rect
            )
            
            # Gera comandos de movimento e targets para o drone
            movement_commands, drone_targets = generate_movement_commands(occupation, movement_threshold)
            
            # Verifica se o alvo foi perdido
            target_lost = "Alvo perdido" in movement_commands
            
            if target_lost:
                frames_without_target += 1
                # Continua com o último movimento válido
                if last_valid_commands and frames_without_target <= 30:  # Mantém por até 30 frames (~1 segundo)
                    movement_commands = last_valid_commands.copy()
                    drone_targets = last_drone_targets.copy()
                    print(f"[TRACK] Mantendo último movimento (frame {frames_without_target}/30)")
                else:
                    # Após 30 frames sem alvo, zoom out para encontrar o alvo
                    movement_commands = ["PROCURANDO_ALVO"]
                    drone_targets = {'move_x': 0, 'move_y': 0, 'zoom_delta': -0.05}
                    print(f"[SEARCH] Zoom out para procurar alvo (frame {frames_without_target})")
            else:
                # Alvo encontrado: salva comandos e reseta contador
                if movement_commands:  # Só salva se houver comandos válidos
                    last_valid_commands = movement_commands.copy()
                    last_drone_targets = drone_targets.copy()
                frames_without_target = 0
            
            # Atualiza targets do drone baseado nos comandos
            if drone_targets['move_x'] != 0 or drone_targets['move_y'] != 0:
                new_target_x = drone.target_x + drone_targets['move_x']
                new_target_y = drone.target_y + drone_targets['move_y']
                drone.set_target(new_target_x, new_target_y)
            
            if drone_targets['zoom_delta'] != 0:
                new_zoom = drone.target_zoom + drone_targets['zoom_delta']
                drone.set_target_zoom(new_zoom)
            
            # Atualiza posição do drone com PID
            velocity_x, velocity_y, velocity_zoom = drone.update()
            
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
            
            # Desenha retângulo da visão do drone primeiro
            if show_drone_view:
                draw_drone_view_rectangle(display_frame, drone)
            
            # Desenha grade APENAS dentro da visão do drone
            if show_grid_overlay:
                draw_grid(display_frame, occupation, drone_view_rect=drone_view_rect)
            
            # Desenha HUD com informações do drone
            draw_hud(display_frame, drone, movement_commands, velocity_x, velocity_y, frame_count)
            
            # Extrai e mostra a visão do drone em janela separada
            if show_drone_view:
                drone_view_frame = extract_drone_view(display_frame, drone)
                cv2.imshow("Drone Camera View", drone_view_frame)
            
            # Mostra frame principal
            cv2.imshow("Rastreamento Customizado", display_frame)
        else:
            # Modo pausado - continua mostrando o último frame
            cv2.imshow("Rastreamento Customizado", display_frame)
        
        # Controles de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            show_grid_overlay = not show_grid_overlay
        elif key == ord("p"):
            paused = not paused
        elif key == ord("d"):
            show_drone_view = not show_drone_view
            print(f"[INFO] Visualização do drone: {'Ativada' if show_drone_view else 'Desativada'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Total de frames processados: {frame_count}")

if __name__ == "__main__":
    main()
