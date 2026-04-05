import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

xp, yp = 0, 0
canvas = None

draw_color = (0, 0, 255)
eraser_mode = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, c = img.shape

    # UI BAR
    cv2.rectangle(img, (0, 0), (w, 80), (40, 40, 40), -1)

    # Color buttons
    cv2.circle(img, (50, 40), 20, (0, 0, 255), -1)
    cv2.circle(img, (150, 40), 20, (0, 255, 0), -1)
    cv2.circle(img, (250, 40), 20, (255, 0, 0), -1)
    cv2.rectangle(img, (320, 20), (380, 60), (200, 200, 200), -1)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            if len(lmList) != 0:
                x1, y1 = lmList[8][1], lmList[8][2]
                x2, y2 = lmList[12][1], lmList[12][2]

                index_up = lmList[8][2] < lmList[6][2]
                middle_up = lmList[12][2] < lmList[10][2]

                # ✌️ Selection Mode
                if index_up and middle_up:
                    xp, yp = 0, 0

                    if y1 < 80:
                        eraser_mode = False

                        if 30 < x1 < 70:
                            draw_color = (0, 0, 255)
                        elif 130 < x1 < 170:
                            draw_color = (0, 255, 0)
                        elif 230 < x1 < 270:
                            draw_color = (255, 0, 0)
                        elif 320 < x1 < 380:
                            eraser_mode = True

                # ☝️ Drawing Mode
                elif index_up and not middle_up:
                    cv2.circle(img, (x1, y1), 10, draw_color, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if eraser_mode:
                        cv2.circle(canvas, (x1, y1), 25, (0, 0, 0), -1)
                    else:
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, 5)

                    xp, yp = x1, y1

                # ✊ Stop
                else:
                    xp, yp = 0, 0

    # Merge canvas
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("AI Virtual Painter PRO", img)

    key = cv2.waitKey(1) & 0xFF
    # 🧼 Clear screen
    if key == ord('c'):
      canvas = np.zeros_like(img)

    # Save image
    if key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()