# Car Counter Project (YOLOv8 + SORT)
Questo progetto implementa un sistema di conteggio automatico dei veicoli (auto, camion, autobus e moto) in un flusso video. Utilizza il modello di rilevamento oggetti YOLOv8 per identificare i veicoli e l'algoritmo di tracciamento SORT (Simple Online and Realtime Tracking) per mantenere traccia degli oggetti rilevati e contarli solo una volta al passaggio di una linea di confine definita.

# Tecnologie Utilizzate
Python: Linguaggio di programmazione principale.

OpenCV (cv2): Gestione del video, mascheramento e visualizzazione.

Ultralytics YOLO: Framework per il rilevamento oggetti (YOLOv8n).

Numpy: Manipolazione di array per il rilevamento e il tracciamento.

cvzone: Utilizzato per semplificare la visualizzazione di rettangoli e testo.

SORT: Implementazione di un tracker multi-oggetto. https://github.com/abewley/sort
