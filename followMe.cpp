#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace dnn;
using namespace std;

// Parâmetros
float confThreshold = 0.3; // confiança mínima
float nmsThreshold = 0.4; // threshold do NMS
int inputWidth = 640;
int inputHeight = 640;

// Variáveis globais
bool grayscaleMode = false;
bool debugMode = false;
int frameCount = 0;

// Função para fazer pós-processamento
void postprocess(Mat& frame, const vector<Mat>& outputs, Mat& displayFrame) {
    Mat out = outputs[0]; 

    int rows, dimensions;

    // Debug do formato inicial
    if (frameCount % 30 == 0) {
        cout << "Formato original da saída: [" << out.rows << ", " << out.cols << "], dims=" << out.dims << endl;
    }

    // Detectar formato da saída e forçar o formato correto
    if (out.dims == 2) {
        if (out.rows == 84 && out.cols == 8400) {
            // Formato [84, 8400] -> transpor para [8400, 84]
            if (frameCount % 30 == 0) {
                cout << "Transpondo de [84, 8400] para [8400, 84]..." << endl;
            }
            cv::transpose(out, out);
            rows = out.rows;        // número de detecções (8400)
            dimensions = out.cols;  // atributos por detecção (84)
        } else if (out.rows == 8400 && out.cols == 84) {
            // Já está no formato correto [8400, 84]
            rows = out.rows;        
            dimensions = out.cols;  
        } else {
            cerr << "Formato 2D inesperado: [" << out.rows << ", " << out.cols << "]" << endl;
            return;
        }
    } else if (out.dims == 3) {
        // Casos possíveis: [1, 8400, 84] ou [1, 84, 8400]
        if (out.size[0] == 1 && out.size[2] == 84) {
            // Formato [1, 8400, 84] - já correto, apenas remove primeira dimensão
            rows = out.size[1];
            dimensions = out.size[2];
            out = out.reshape(0, rows); // Remove primeira dimensão -> [8400, 84]
        } else if (out.size[0] == 1 && out.size[1] == 84 && out.size[2] == 8400) {
            // Formato [1, 84, 8400] - precisa transpor
            if (frameCount % 30 == 0) {
                cout << "Tratando formato [1, 84, 8400]..." << endl;
            }
            out = out.reshape(0, out.size[1]); // Remove primeira dimensão -> [84, 8400]
            cv::transpose(out, out); // Transpõe para [8400, 84]
            rows = out.rows;
            dimensions = out.cols;
        } else {
            cerr << "Formato 3D inesperado: [" << out.size[0] << ", " << out.size[1] << ", " << out.size[2] << "]" << endl;
            return;
        }
    } else if (out.dims == 4) {
        // Caso [1, 84, 8400, 1] ou similar
        if (out.size[0] == 1 && out.size[1] == 84 && out.size[3] == 1) {
            Mat temp = out.reshape(0, out.size[1] * out.size[2]); // [84*8400]
            temp = temp.reshape(0, out.size[1]); // [84, 8400]
            cv::transpose(temp, out); // [8400, 84]
            rows = out.rows;
            dimensions = out.cols;
        } else {
            cerr << "Formato 4D inesperado: [" << out.size[0] << ", " << out.size[1] << ", " << out.size[2] << ", " << out.size[3] << "]" << endl;
            return;
        }
    } else {
        cerr << "Formato inesperado da saída: dims=" << out.dims << endl;
        return;
    }
    
    // Verifica se temos o formato esperado
    if (dimensions != 84) {
        cerr << "ERRO: Esperado 84 dimensões, obtido " << dimensions << endl;
        cerr << "Formato final: [" << rows << ", " << dimensions << "]" << endl;
        return;
    }

    const float* data = (float*)out.data;

    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    int detectionsAboveThreshold = 0;
    int personCandidates = 0;

    // Log de debug do formato da saída a cada 30 frames
    if (frameCount % 30 == 0) {
        cout << "Frame " << frameCount << ": Formato da saída: [" << rows << ", " << dimensions << "]" << endl;
    }

    for (int i = 0; i < rows; i++) {
        // YOLOv8 format: [cx, cy, w, h, class0_score, class1_score, ..., class79_score]
        // As classes começam no índice 4
        Mat scores(1, dimensions - 4, CV_32F, (void*)(data + i * dimensions + 4));
        Point classIdPoint;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &classIdPoint);

        // Debug: contar detecções acima de um threshold baixo
        if (maxClassScore > 0.1) {
            detectionsAboveThreshold++;
            
            // Mostra algumas detecções para debug
            if (debugMode && detectionsAboveThreshold <= 5 && frameCount % 30 == 0) {
                cout << "  Detecção " << i << ": classe=" << classIdPoint.x 
                     << ", conf=" << fixed << setprecision(3) << maxClassScore << endl;
            }
        }

        // Filtra apenas pessoas (classe 0) com confiança mínima - reduzindo threshold para debug
        if (maxClassScore > 0.25 && classIdPoint.x == 0) {
            personCandidates++;
            
            // Converte coordenadas normalizadas para pixels
            // YOLOv8 retorna coordenadas já em escala da imagem de entrada (640x640)
            float cx = data[i * dimensions + 0];
            float cy = data[i * dimensions + 1];
            float w = data[i * dimensions + 2];
            float h = data[i * dimensions + 3];
            
            // Converte para o tamanho real da imagem
            int x = (int)((cx - w/2) * frame.cols / 640.0);
            int y = (int)((cy - h/2) * frame.rows / 640.0);
            int width = (int)(w * frame.cols / 640.0);
            int height = (int)(h * frame.rows / 640.0);
            
            // Garante coordenadas dentro dos limites
            x = max(0, x);
            y = max(0, y);
            width = min(width, frame.cols - x);
            height = min(height, frame.rows - y);

            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)maxClassScore);
            boxes.push_back(Rect(x, y, width, height));
            
            // Log detalhado das pessoas detectadas
            if (debugMode || frameCount % 30 == 0) {
                cout << "  PESSOA detectada " << personCandidates << ": conf=" 
                     << fixed << setprecision(3) << maxClassScore 
                     << ", pos=(" << x << "," << y << "), tam=(" << width << "," << height << ")" << endl;
            }
        }
    }
    
    // Log do resumo a cada 30 frames
    if (frameCount % 30 == 0) {
        cout << "  Detecções acima de 0.1: " << detectionsAboveThreshold << endl;
        cout << "  Candidatos a pessoa: " << personCandidates << endl;
    }

    // Copia o frame de deteccao para o frame de exibição
    displayFrame = grayscaleMode ? frame.clone() : frame.clone();
    
    // NMS para remover sobreposição
    vector<int> indices;
    int finalDetections = 0;
    
    if (!boxes.empty()) {
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        
        for (int idx : indices) {
            Rect box = boxes[idx];
            float confidence = confidences[idx];
            
            // Cor da caixa: verde para colorido, branco para P&B
            Scalar boxColor = grayscaleMode ? Scalar(255, 255, 255) : Scalar(0, 255, 0);
            Scalar textBgColor = grayscaleMode ? Scalar(200, 200, 200) : Scalar(0, 255, 0);
            Scalar textColor = Scalar(0, 0, 0);
            
            rectangle(displayFrame, box, boxColor, 2);
            
            // Desenha texto com fundo
            string label = "Pessoa: " + to_string(confidence).substr(0, 4);
            int baseline;
            Size textSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            
            rectangle(displayFrame, 
                     Point(box.x, box.y - textSize.height - baseline),
                     Point(box.x + textSize.width, box.y),
                     textBgColor, -1);
            
            putText(displayFrame, label, Point(box.x, box.y - baseline),
                    FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
            
            finalDetections++;
        }
    }
    
    // Adiciona informações na tela
    string modeText = grayscaleMode ? "P&B" : "Colorido";
    string infoText = "Pessoas: " + to_string(finalDetections) + 
                     " | Frame: " + to_string(frameCount) + 
                     " | Modo: " + modeText;
    
    Scalar infoColor = grayscaleMode ? Scalar(200, 200, 200) : Scalar(255, 255, 255);
    putText(displayFrame, infoText, Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 0.7, infoColor, 2);
    
    putText(displayFrame, "Controles: 'g'-Alternar | 'c'-Colorido | 'b'-P&B | 'q'-Sair", 
            Point(10, displayFrame.rows - 10), 
            FONT_HERSHEY_SIMPLEX, 0.4, infoColor, 1);
    
    // Log no console com informações de detecção
    if (frameCount % 30 == 0) {
        cout << "  Pessoas após NMS: " << finalDetections << endl;
        cout << "Frame " << frameCount << " processado - Modo: " << modeText << endl;
        cout << "----------------------------------------" << endl;
    }
    
    // Log imediato quando detecta pessoas
    if (finalDetections > 0) {
        static int lastLogFrame = 0;
        if (frameCount - lastLogFrame >= 30) {  // Log a cada 30 frames quando há detecções
            cout << "✓ Frame " << frameCount << ": " << finalDetections 
                 << " pessoa(s) detectada(s) - Modo: " << modeText << endl;
            lastLogFrame = frameCount;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <modelo.onnx>" << endl;
        return -1;
    }

    cout << "Carregando modelo YOLOv8..." << endl;
    
    // Carregar modelo
    Net net = cv::dnn::readNetFromONNX(argv[1]);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    cout << "Modelo carregado com sucesso!" << endl;

    // Abrir câmera
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir a câmera!" << endl;
        return -1;
    }
    
    // Configurações da câmera
    cap.set(CAP_PROP_FRAME_WIDTH, 720);
    cap.set(CAP_PROP_FRAME_HEIGHT, 1280);
    
    cout << "=== CONTROLES ===" << endl;
    cout << "'q' - Sair" << endl;
    cout << "'g' - Alternar entre Colorido/Preto e Branco" << endl;
    cout << "'c' - Modo Colorido" << endl;
    cout << "'b' - Modo Preto e Branco" << endl;
    cout << "'d' - Mostrar debug detalhado" << endl;
    cout << "=================" << endl;
    cout << "Iniciando detecção de pessoas..." << endl;

    while (true) {
        Mat frame, originalFrame, detectionFrame, displayFrame;
        cap >> frame;
        if (frame.empty()) break;
        
        frameCount++;
        originalFrame = frame.clone();
        
        // Frame usado para detecção (pode ser P&B)
        detectionFrame = frame.clone();
        if (grayscaleMode) {
            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            cvtColor(gray, detectionFrame, COLOR_GRAY2BGR);
        }

        // Criar blob
        Mat blob = blobFromImage(detectionFrame, 1/255.0, Size(inputWidth, inputHeight), Scalar(), true, false);
        net.setInput(blob);

        // Inferência
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // Pós-processamento
        postprocess(detectionFrame, outputs, displayFrame);

        imshow("YOLOv8 - Detecção de Pessoas", displayFrame);

        char c = (char)waitKey(1);
        if (c == 'q') {
            cout << "Encerrando..." << endl;
            break;
        }
        else if (c == 'g') {
            grayscaleMode = !grayscaleMode;
            string newMode = grayscaleMode ? "Preto e Branco" : "Colorido";
            cout << "Modo alterado para: " << newMode << endl;
        }
        else if (c == 'c') {
            if (grayscaleMode) {
                grayscaleMode = false;
                cout << "Modo alterado para: Colorido" << endl;
            }
        }
        else if (c == 'b') {
            if (!grayscaleMode) {
                grayscaleMode = true;
                cout << "Modo alterado para: Preto e Branco" << endl;
            }
        }
        else if (c == 'd') {
            debugMode = !debugMode;
            cout << "Modo debug " << (debugMode ? "ATIVADO" : "DESATIVADO") << endl;
            if (debugMode) {
                cout << "Debug ativo: mostrará detecções detalhadas no console" << endl;
            }
        }
    }

    cap.release();
    destroyAllWindows();
    cout << "Programa finalizado." << endl;
    return 0;
}
