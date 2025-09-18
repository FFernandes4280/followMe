import cv2
import numpy as np

print("Carregando modelo YOLOv8...")
# Carrega o modelo YOLOv8 em formato ONNX
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")
print("Modelo carregado com sucesso!")

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera")
    exit()

# Configurações da webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Classes do COCO (apenas índice 0 é "person")
class_names = ["person"]
print("Iniciando detecção de pessoas...")
print("Pressione 'q' para sair")
print("Pressione 'g' para alternar entre colorido/preto e branco")

frame_count = 0
grayscale_mode = False  # Flag para controlar modo preto e branco

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image_height, image_width = frame.shape[:2]
    
    # Cria uma cópia do frame original para detecção
    detection_frame = frame.copy()
    
    # Se modo preto e branco estiver ativo, converte o frame de detecção
    if grayscale_mode:
        detection_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_GRAY2BGR)  # Volta para 3 canais para o YOLO

    # Redimensiona para 640x640 (tamanho padrão YOLOv8)
    blob = cv2.dnn.blobFromImage(detection_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # Executa a rede
    outputs = net.forward()
    outputs = outputs[0]  # Remove dimensão extra se existir
    
    # Transpõe se necessário para ter formato (num_detections, num_values)
    if outputs.shape[0] == 84:  # Se está no formato (84, 8400)
        outputs = outputs.T  # Transpõe para (8400, 84)

    rows = outputs.shape[0]
    boxes, confidences, class_ids = [], [], []

    for i in range(rows):
        detection = outputs[i]
        
        # YOLOv8 format: [cx, cy, w, h, class0_score, class1_score, ..., class79_score]
        cx, cy, w, h = detection[0:4]
        scores = detection[4:]  # Scores para todas as 80 classes
        
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        # Filtra apenas pessoas (classe 0) com confiança mínima
        if class_id == 0 and confidence > 0.3:
            # Converte coordenadas do centro para coordenadas do canto superior esquerdo
            x = int((cx - w/2) * image_width / 640)
            y = int((cy - h/2) * image_height / 640)
            width = int(w * image_width / 640)
            height = int(h * image_height / 640)
            
            # Garante que as coordenadas estão dentro dos limites da imagem
            x = max(0, x)
            y = max(0, y)
            width = min(width, image_width - x)
            height = min(height, image_height - y)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # NMS para remover caixas duplicadas
    final_detections = 0
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        
        if len(indices) > 0:
            # Corrige o acesso aos índices para diferentes versões do OpenCV
            if isinstance(indices, np.ndarray) and indices.ndim == 2:
                indices = indices.flatten()
            
            for i in indices:
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Desenha o texto com fundo para melhor visibilidade
                label = f"Pessoa: {confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y), (0, 255, 0), -1)
                cv2.putText(frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                final_detections += 1
    
    # Prepara o frame para exibição (pode ser diferente do frame de detecção)
    display_frame = detection_frame.copy() if grayscale_mode else frame.copy()
    
    # Redesenha as detecções no frame de exibição
    if len(boxes) > 0 and len(indices) > 0:
        if isinstance(indices, np.ndarray) and indices.ndim == 2:
            indices = indices.flatten()
        
        for i in indices:
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Desenha o texto com fundo para melhor visibilidade
            label = f"Pessoa: {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x, y - text_height - baseline), (x + text_width, y), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Adiciona informações na tela
    mode_text = "P&B" if grayscale_mode else "Colorido"
    info_text = f"Pessoas: {final_detections} | Frame: {frame_count} | Modo: {mode_text}"
    cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "Pressione 'g' para alternar modo | 'q' para sair", (10, image_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Imprime informações no console a cada 30 frames se houver detecções
    if final_detections > 0 and frame_count % 30 == 0:
        print(f"Frame {frame_count}: {final_detections} pessoa(s) detectada(s) - Modo: {mode_text}")

    cv2.imshow("YOLOv8 - Pessoas", display_frame)

    # Verifica teclas pressionadas
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        grayscale_mode = not grayscale_mode
        mode_text = "Preto e Branco" if grayscale_mode else "Colorido"
        print(f"Modo alterado para: {mode_text}")

cap.release()
cv2.destroyAllWindows()
