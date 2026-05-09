import os
import time

import cv2
from ultralytics import YOLO

# 1. Загружаем модель
model = YOLO("models/yolo26x.pt")

# 2. Инициализируем камеру (0 — индекс стандартной веб-камеры)
cap = cv2.VideoCapture(0)
prev_time = 0
cap.set(cv2.CAP_PROP_FPS, 60)

current_dir = os.path.dirname(os.path.abspath(__file__))
tracker_path = os.path.join(current_dir, "my_tracker.yaml")

while cap.isOpened():
    success, frame = cap.read()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if success:
        # 3. Запускаем трекинг для текущего кадра
        # persist=True говорит модели, что это поток, и нужно сохранять ID между кадрами
        results = model.track(frame, persist=True, classes=[0], conf=0.6, tracker=tracker_path, verbose=False)

        # 4. Визуализируем результаты на кадре
        annotated_frame = results[0].plot()

        # 5. Показываем результат
        cv2.imshow("YOLO Tracking (Kalman Filter)", annotated_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()