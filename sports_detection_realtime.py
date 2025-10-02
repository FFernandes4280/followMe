#!/usr/bin/env python33
"""
Script de Detecção em Tempo Real para Pessoas em Esportes
Baseado no followMe.py, este script detecta pessoas fazendo esportes em tempo real.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO
import argparse

class SportsDetectionRealtime:
    def __init__(self, model_path="sports_detection_best.pt", confidence_threshold=0.3):
        """
        Inicializa o sistema de detecção em tempo real
        
        Args:
            model_path: Caminho para o modelo treinado
            confidence_threshold: Threshold de confiança para detecções
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cap = None
        self.load_model()
        
    def load_model(self):
        """Carrega o modelo treinado"""
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                print(f"✅ Modelo carregado: {self.model_path}")
            else:
                # Fallback para modelo padrão
                print(f"⚠️  Modelo não encontrado: {self.model_path}")
                print("   Usando modelo YOLOv8 padrão...")
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def setup_camera(self, camera_id=0, width=720, height=1280):
        """
        Configura a câmera
        
        Args:
            camera_id: ID da câmera
            width: Largura da imagem
            height: Altura da imagem
        """
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            print("❌ Erro: Não foi possível abrir a câmera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print(f"📹 Câmera configurada: {width}x{height}")
        return True
    
    def calculate_grid_occupation(self, boxes, indices, image_width, image_height):
        """
        Calcula ocupação de cada quadrado da grade 3x3
        Baseado no código original do followMe.py
        """
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
    
    def generate_movement_commands(self, occupation, threshold=0.1):
        """
        Gera comandos de movimento baseado na ocupação da grade
        Baseado no código original do followMe.py
        """
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
    
    def draw_grid(self, frame, image_width, image_height):
        """
        Desenha a grade 3x3 na imagem
        
        Args:
            frame: Frame da imagem
            image_width: Largura da imagem
            image_height: Altura da imagem
        """
        grid_width = image_width // 3
        grid_height = image_height // 3
        
        # Desenha linhas verticais
        for i in range(1, 3):
            x = i * grid_width
            cv2.line(frame, (x, 0), (x, image_height), (255, 255, 255), 1)
        
        # Desenha linhas horizontais
        for i in range(1, 3):
            y = i * grid_height
            cv2.line(frame, (0, y), (image_width, y), (255, 255, 255), 1)
    
    def draw_occupation_info(self, frame, occupation, image_width, image_height):
        """
        Desenha informações de ocupação na imagem
        
        Args:
            frame: Frame da imagem
            occupation: Matriz de ocupação 3x3
            image_width: Largura da imagem
            image_height: Altura da imagem
        """
        grid_width = image_width // 3
        grid_height = image_height // 3
        
        for row in range(3):
            for col in range(3):
                x = col * grid_width
                y = row * grid_height
                
                # Cor baseada na ocupação
                intensity = int(occupation[row, col] * 255)
                color = (0, intensity, 0) if intensity > 0 else (50, 50, 50)
                
                # Desenha retângulo semi-transparente
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + grid_width, y + grid_height), color, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Texto com porcentagem
                text = f"{occupation[row, col]:.1%}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = x + (grid_width - text_size[0]) // 2
                text_y = y + (grid_height + text_size[1]) // 2
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_detection(self, show_grid=True, show_occupation=True, save_video=False):
        """
        Executa detecção em tempo real
        
        Args:
            show_grid: Se deve mostrar a grade 3x3
            show_occupation: Se deve mostrar informações de ocupação
            save_video: Se deve salvar o vídeo
        """
        if not self.cap:
            print("❌ Câmera não configurada!")
            return
        
        print("🚀 Iniciando detecção em tempo real...")
        print("   Pressione 'q' para sair")
        print("   Pressione 'g' para alternar grade")
        print("   Pressione 'o' para alternar ocupação")
        print("   Pressione 's' para salvar frame")
        
        # Configuração para salvar vídeo
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter('sports_detection_output.avi', fourcc, 20.0, (720, 1280))
        
        frame_count = 0
        movement_threshold = 0.15
        print_occupation_grid = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                image_height, image_width = frame.shape[:2]
                
                # Executa detecção
                start_time = time.time()
                results = self.model(frame, conf=self.confidence_threshold)
                inference_time = time.time() - start_time
                
                # Processa resultados
                detections = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = box.astype(int)
                        w, h = x2 - x1, y2 - y1
                        detections.append([x1, y1, w, h])
                        
                        # Desenha bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calcula ocupação da grade
                occupation = self.calculate_grid_occupation(detections, 
                                                          list(range(len(detections))), 
                                                          image_width, image_height)
                
                # Gera comandos de movimento
                movement_commands = self.generate_movement_commands(occupation, movement_threshold)
                
                # Desenha grade se habilitada
                if show_grid:
                    self.draw_grid(frame, image_width, image_height)
                
                # Desenha informações de ocupação se habilitada
                if show_occupation:
                    self.draw_occupation_info(frame, occupation, image_width, image_height)
                
                # Exibe comandos de movimento
                if movement_commands:
                    for i, cmd in enumerate(movement_commands):
                        cv2.putText(frame, f"CMD: {cmd}", (10, 30 + i*25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Exibe informações de performance
                fps = 1.0 / inference_time if inference_time > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (image_width - 100, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Detections: {len(detections)}", (image_width - 150, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Exibe grade de ocupação no terminal se habilitada
                if print_occupation_grid and frame_count % 30 == 0:  # A cada 30 frames
                    print("Grade de ocupação:")
                    for row in range(3):
                        row_str = ""
                        for col in range(3):
                            row_str += f"{occupation[row, col]:6.1%} "
                        print(f"  [{row_str}]")
                    print("-" * 50)
                
                # Salva vídeo se habilitado
                if video_writer:
                    video_writer.write(frame)
                
                # Exibe frame
                cv2.imshow("Detecção de Pessoas em Esportes", frame)
                
                # Processa teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("g"):
                    show_grid = not show_grid
                    print(f"Grade: {'ON' if show_grid else 'OFF'}")
                elif key == ord("o"):
                    show_occupation = not show_occupation
                    print(f"Ocupação: {'ON' if show_occupation else 'OFF'}")
                elif key == ord("s"):
                    filename = f"sports_detection_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame salvo: {filename}")
                elif key == ord("p"):
                    print_occupation_grid = not print_occupation_grid
                    print(f"Impressão da grade: {'ON' if print_occupation_grid else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⏹️  Detecção interrompida pelo usuário")
        
        finally:
            # Limpa recursos
            if video_writer:
                video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Detecção finalizada")
    
    def detect_from_image(self, image_path, output_path=None):
        """
        Detecta pessoas em uma imagem estática
        
        Args:
            image_path: Caminho para a imagem
            output_path: Caminho para salvar resultado (opcional)
        """
        print(f"🖼️  Detectando em: {image_path}")
        
        # Carrega imagem
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Erro ao carregar imagem: {image_path}")
            return
        
        image_height, image_width = frame.shape[:2]
        
        # Executa detecção
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Processa resultados
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                detections.append([x1, y1, w, h])
                
                # Desenha bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calcula ocupação da grade
        occupation = self.calculate_grid_occupation(detections, 
                                                  list(range(len(detections))), 
                                                  image_width, image_height)
        
        # Desenha grade e ocupação
        self.draw_grid(frame, image_width, image_height)
        self.draw_occupation_info(frame, occupation, image_width, image_height)
        
        # Salva resultado
        if output_path is None:
            output_path = f"result_{Path(image_path).name}"
        
        cv2.imwrite(output_path, frame)
        print(f"✅ Resultado salvo em: {output_path}")
        print(f"📊 Detecções encontradas: {len(detections)}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Detecção de Pessoas em Esportes em Tempo Real")
    parser.add_argument("--model", default="sports_detection_best.pt", 
                       help="Caminho para o modelo treinado")
    parser.add_argument("--confidence", type=float, default=0.3, 
                       help="Threshold de confiança para detecções")
    parser.add_argument("--camera", type=int, default=0, 
                       help="ID da câmera")
    parser.add_argument("--image", type=str, 
                       help="Caminho para imagem estática (opcional)")
    parser.add_argument("--save-video", action="store_true", 
                       help="Salvar vídeo de saída")
    parser.add_argument("--no-grid", action="store_true", 
                       help="Não mostrar grade 3x3")
    parser.add_argument("--no-occupation", action="store_true", 
                       help="Não mostrar informações de ocupação")
    
    args = parser.parse_args()
    
    print("🏃‍♂️ Sistema de Detecção de Pessoas em Esportes")
    print("=" * 50)
    
    # Inicializa detector
    detector = SportsDetectionRealtime(args.model, args.confidence)
    
    if args.image:
        # Detecção em imagem estática
        detector.detect_from_image(args.image)
    else:
        # Detecção em tempo real
        if detector.setup_camera(args.camera):
            detector.run_detection(
                show_grid=not args.no_grid,
                show_occupation=not args.no_occupation,
                save_video=args.save_video
            )

if __name__ == "__main__":
    main()
