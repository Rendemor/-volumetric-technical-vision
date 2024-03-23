import mediapipe as mp
import cv2
import numpy as np
import pyfirmata

mpHands = mp.solutions.hands  # подключаем раздел распознавания рук
hands = mpHands.Hands(1)  # создаем объект класса "руки"
mpDraw = mp.solutions.drawing_utils  # подключаем инструменты для рисования
distanceBetweenCameras = 21  # расстояние между камерами
thresholdValueOffset = 20  # порог смещения пальца относительно центра камеры для всех осей
st = 0

# получение видеопотока с камер
leftCamera = cv2.VideoCapture(0)
rightCamera = cv2.VideoCapture(1)

# переменная для детекции состояния системы (разблокирована или нет)
unlockingRightCam = False
unlockingLeftCam = False

# несколько доп. параметров, по типу цвет, толщина текста и т.д.
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
bold = 3
boldM = 2

board = pyfirmata.Arduino('COM6')

# обрабатываю изображение нейронной сетью mediapipe. Получаю координаты (x, y) каждой точки (20 шт) на руке
def neuralNetwork(cam):
    imgRGBCam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # смена режима изображения для распознавания рук
    resultsCam = hands.process(imgRGBCam)  # получаем результат распознавания
    cord = {0: None}
    if resultsCam.multi_hand_landmarks:
        cord.clear()
        for handLms in resultsCam.multi_hand_landmarks:
            mpDraw.draw_landmarks(cam, handLms, mpHands.HAND_CONNECTIONS)
            for id, point in enumerate(handLms.landmark):
                w, h, c = cam.shape  # width, height, color
                Xcam, Ycam = int(point.x * h), int(point.y * w)  # определяем координаты базовых точек
                cord[id] = [Xcam, Ycam]
                if id == 8:
                    cv2.circle(cam, (Xcam, Ycam), 10, (0, 255, 0), -1)
    print(cord)
    return cam, cord


# функция-активатор системы. Накладывается на каждую камеру. Это помогает избежать случайных срабатываний,
# чтобы активировать систему нужно навести указательным пальцем на фигуру на обоих изображениях
def activate(cam, cord):
    act = False
    X_Rect, Y_Rect = cam.shape[1], cam.shape[0]
    X1 = int(X_Rect / 2 - X_Rect / 10)
    Y1 = int(Y_Rect / 2 - Y_Rect / 10)
    X2 = int(X_Rect / 2 + X_Rect / 10)
    Y2 = int(Y_Rect / 2 + Y_Rect / 10)
    cv2.rectangle(cam, (X1, Y1), (X2, Y2), [0, 0, 255], -1)
    if X1 < cord[0] < X2 and Y1 < cord[1] < Y2:
        act = True
    return cam, act


# нанесение на изображение осевых линий камер
def lines(cam):
    X_Rect, Y_Rect = cam.shape[1], cam.shape[0]
    X1, Y1 = int(X_Rect / 2), 0
    X2, Y2 = int(X_Rect / 2), Y_Rect
    X3, Y3 = 0, int(Y_Rect / 2)
    X4, Y4 = X_Rect, int(Y_Rect / 2)
    cv2.line(cam, (X1, Y1), (X2, Y2), (255, 0, 0), 1)
    cv2.line(cam, (X3, Y3), (X4, Y4), (255, 0, 0), 1)
    return cam

# смещение указательного пальца по высоте (y) относительно левой камеры
def miroLH(cord, pix_all):
    offset = {}
    for i in range(len(cord)):
        if cord[i][0] > pix_all / 2:
            offset[i] = (cord[i][0] - pix_all / 2)  # угол с плюсом
        elif cord[i][0] < pix_all / 2:
            offset[i] = (pix_all / 2 - cord[i][0]) * (-1)  # угол с минусом
        elif cord[i][0] == pix_all / 2:
            offset[i] = 0  # угол ноль
    return offset


# смещение указательного пальца по высоте (y) относительно правой камеры
def miroRH(cord, pix_all):
    offset = {}
    for i in range(len(cord)):
        if cord[i][0] > pix_all / 2:
            offset[i] = (cord[i][0] - pix_all / 2) * (-1)  # угол с плюсом
        elif cord[i][0] < pix_all / 2:
            offset[i] = (pix_all / 2 - cord[i][0])  # угол с минусом
        elif cord[i][0] == pix_all / 2:
            offset[i] = 0  # угол ноль
    return offset


# смещение указательного пальца по щирине (x) относительно левой камеры
def miroLV(cord, pix_all):
    offset = {}
    for i in range(len(cord)):
        if cord[i][1] > pix_all / 2:
            offset[i] = (cord[i][1] - pix_all / 2)  # угол с плюсом
        elif cord[i][1] < pix_all / 2:
            offset[i] = (pix_all / 2 - cord[i][1]) * (-1)  # угол с минусом
        elif cord[i][1] == pix_all / 2:
            offset[i] = 0  # угол ноль
    return offset


# смещение указательного пальца по щирине (x) относительно правой камеры
def miroRV(cord, pix_all):
    offset = {}
    for i in range(len(cord)):
        if cord[i][1] > pix_all / 2:
            offset[i] = (cord[i][1] - pix_all / 2)  # угол с плюсом
        elif cord[i][1] < pix_all / 2:
            offset[i] = (pix_all / 2 - cord[i][1]) * (-1)  # угол с минусом
        elif cord[i][1] == pix_all / 2:
            offset[i] = 0  # угол ноль
    return offset


# определяет смещение указательного пальца в градусах по горизонтали и вертикали. Эта функция вызывается для кадого
# параметра каждой камеры
def angle(omega, katet, pix):
    omega /= 2
    katet /= 2
    angle = {}
    for i in range(len(pix)):
        tang = np.tan(omega * np.pi / 180)  # определяем тангенс угла
        Hig = katet / tang  # определили катет
        angle[i] = np.arctan(pix[i] / Hig) * 180 / np.pi  # определяем угол и переводим в градусы
    return angle


# рассчёт глубины (расстояние, на котором объект находится от камеры (левой, т.к. она была выбрана в качестве главной))
def longH(alpha_L, alpha_R, long_B):  # функция ()
    # alpha_L - угол, на котором находится объект относительно левой камеры
    # alpha_R - угол, на котором находится объект относительно правой камеры
    # long_B - расстояние между камерами (мм)
    nLong_H = {}
    for i in range(len(alpha_L)):
        beta_L = 90 - alpha_L[i]  # угол треугольника от левой камеры
        beta_R = 90 - alpha_R[i]  # угол треугольника от правой камеры
        beta_H = 180 - beta_R - beta_L  # угол вершины треугольника (третий угол треугольника)
        sin_H = np.sin(beta_H * np.pi / 180)  # находим синус угла вершины
        sin_L = np.sin(beta_L * np.pi / 180)  # находим синус левого угла
        sin_R = np.sin(beta_R * np.pi / 180)  # находим синус правого угла

        long_L = long_B * sin_R / sin_H  # находим сторону напротив правого угла
        long_R = long_B * sin_L / sin_H  # находим сторону напротив левого угла

        long_H1 = long_L * sin_L  # высота слева
        long_H2 = long_R * sin_R  # высота справа
        nLong_H[i] = (long_H1 + long_H2) / 2  # определяем среднее значение

    return nLong_H  # возвращаем результат

def camY(depth):
    Y = {}
    for i in range(len(depth)):
        Y[i] = 70 - depth[i]
    return Y


# смещение по оси X (вправо/влево) в см. относительно левой камеры
def camX(depth, angle):
    X = {}
    for i in range(len(depth)):
        X[i] = depth[i] * np.tan(angle[i] * np.pi / 180)
    return X

# смещение по оси Z (вверх/вниз) в см. относительно левой камеры
def camZ(depth, angle):
    Z = {}
    for i in range(len(depth)):
        Z[i] = depth[i] * np.tan(angle[i] * np.pi / 180)
    return Z


# 1 - вправо
# 2 - влево
# 3 - вперёд
# 4 - назад
# 5 - вниз
# 6 - вверх
# 7 - захват
# 8 - поворот
def definitionMovement(deltaX, deltaY, deltaZ, gest):
    status = 0
    if abs(deltaX) > thresholdValueOffset or abs(deltaY) > thresholdValueOffset or abs(deltaZ) > thresholdValueOffset:
        if abs(deltaX) > abs(deltaY) and abs(deltaX) > abs(deltaZ):
            if deltaX > 0:
                status = 1
            else:
                status = 2

        elif abs(deltaY) > abs(deltaX) and abs(deltaY) > abs(deltaZ):
            if deltaY > 0:
                status = 3
            else:
                status = 4

        if abs(deltaZ) > abs(deltaX) and abs(deltaZ) > abs(deltaY):
            if deltaZ > 0:
                status = 5
            else:
                status = 6
    # elif gest == "Catch":
    #     status = 7
    # elif gest == "Turn":
    #     status = 8
    return status


# вывод сетки, чтобы было видно, какое именно движение сейчас
def paintGryd(cam, st, gest):
    X_Rect, Y_rect = cam.shape[1], cam.shape[0]

    X0 = int(X_Rect / 2)
    Y0 = int(Y_rect / 2)
    X1 = X0
    Y1 = Y0 - 200
    X2 = X0 + 175
    Y2 = Y0 - 105
    X3 = X0 + 175
    Y3 = Y0 + 105
    X4 = X0
    Y4 = Y0 + 200
    X5 = X0 - 175
    Y5 = Y0 + 105
    X6 = X0 - 175
    Y6 = Y0 - 105

    cv2.arrowedLine(cam, (X0, Y0), (X1, Y1), black, bold)
    cv2.arrowedLine(cam, (X0, Y0), (X2, Y2), black, bold)
    cv2.arrowedLine(cam, (X0, Y0), (X3, Y3), black, bold)
    cv2.arrowedLine(cam, (X0, Y0), (X4, Y4), black, bold)
    cv2.arrowedLine(cam, (X0, Y0), (X5, Y5), black, bold)
    cv2.arrowedLine(cam, (X0, Y0), (X6, Y6), black, bold)

    cv2.putText(cam, "+Z, Up", (X1 + 20, Y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, boldM)
    cv2.putText(cam, "-Y, Back", (X2 + 20, Y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, boldM)
    cv2.putText(cam, "+X, Right", (X3 + 20, Y3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, boldM)
    cv2.putText(cam, "-Z, Down", (X4 + 20, Y4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, boldM)
    cv2.putText(cam, "+Y, Forward", (X5 + 20, Y5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, boldM)
    cv2.putText(cam, "-X, Left", (X6 + 20, Y6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, black, boldM)

    # поворот
    point1 = np.array([[X5, Y5 + 50], [X5 - 50, Y5 + 50], [X5 - 50, Y5 + 30]])
    point2 = np.array([[X5, Y5 - 50], [X5 - 50, Y5 - 50], [X5 - 50, Y5 - 30]])
    cv2.polylines(cam, [point1], 0, black, bold)
    cv2.polylines(cam, [point2], 0, black, bold)

    # Захват
    cv2.rectangle(cam, (5, 5), (100 + 5, 60 + 5), black, bold)
    cv2.line(cam, (25, 65), (25, 120), black, bold)
    cv2.line(cam, (25, 120), (50, 120), black, bold)
    cv2.line(cam, (80, 65), (80, 120), black, bold)
    cv2.line(cam, (80, 120), (55, 120), black, bold)

    cv2.rectangle(cam, (5, 150), (105, 150 + 60), black, bold)
    cv2.line(cam, (25 + 15, 150 + 60), (5 + 15, 150 + 60 + 85), black, bold)
    cv2.line(cam, (5 + 15, 150 + 60 + 85), (25 + 15, 150 + 60 + 100), black, bold)
    cv2.line(cam, (80 - 15, 150 + 60), (105 - 15, 150 + 60 + 85), black, bold)
    cv2.line(cam, (105 - 15, 150 + 60 + 85), (80 - 15, 150 + 60 + 100), black, bold)

    # 1 - вправо, 2 - влево, 3 - вперёд, 4 - назад, 5 - вниз, 6 - вверх
    if st == 1:
        cv2.arrowedLine(cam, (X0, Y0), (X3, Y3), blue, bold)
        cv2.putText(cam, "+X, Right", (X3 + 20, Y3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, boldM)
    elif st == 2:
        cv2.arrowedLine(cam, (X0, Y0), (X6, Y6), blue, bold)
        cv2.putText(cam, "-X, Left", (X6 + 20, Y6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, boldM)
    elif st == 3:
        cv2.arrowedLine(cam, (X0, Y0), (X5, Y5), blue, bold)
        cv2.putText(cam, "+Y, Forward", (X5 + 20, Y5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, boldM)
    elif st == 4:
        cv2.arrowedLine(cam, (X0, Y0), (X2, Y2), blue, bold)
        cv2.putText(cam, "-Y, Back", (X2 + 20, Y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, boldM)
    elif st == 5:
        cv2.arrowedLine(cam, (X0, Y0), (X4, Y4), blue, bold)
        cv2.putText(cam, "-Z, Down", (X4 + 20, Y4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, boldM)
    elif st == 6:
        cv2.arrowedLine(cam, (X0, Y0), (X1, Y1), blue, bold)
        cv2.putText(cam, "+Z, Up", (X1 + 20, Y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blue, boldM)

    if gest == "Catch":
        cv2.rectangle(cam, (5, 5), (100 + 5, 60 + 5), blue, bold)
        cv2.line(cam, (25, 65), (25, 120), blue, bold)
        cv2.line(cam, (25, 120), (50, 120), blue, bold)
        cv2.line(cam, (80, 65), (80, 120), blue, bold)
        cv2.line(cam, (80, 120), (55, 120), blue, bold)
    elif gest == "No Gesture":
        cv2.rectangle(cam, (5, 150), (105, 150 + 60), blue, bold)
        cv2.line(cam, (25 + 15, 150 + 60), (5 + 15, 150 + 60 + 85), blue, bold)
        cv2.line(cam, (5 + 15, 150 + 60 + 85), (25 + 15, 150 + 60 + 100), blue, bold)
        cv2.line(cam, (80 - 15, 150 + 60), (105 - 15, 150 + 60 + 85), blue, bold)
        cv2.line(cam, (105 - 15, 150 + 60 + 85), (80 - 15, 150 + 60 + 100), blue, bold)
    elif gest == "Turn":
        cv2.polylines(cam, [point1], 0, blue, bold)
    return cam

# управление arduino через python с пк
# 1 - вправо
# 2 - влево
# 3 - вперёд
# 4 - назад
# 5 - вниз
# 6 - вверх
# 7 - захват
# 8 - поворот
# 7 и 8 нет, так как нужные двигатели пока что отсутствуют
def go(st):
    if st == 0:
        board.digital[2].write(0)
        board.digital[3].write(0)
        board.digital[4].write(0)
        board.digital[5].write(0)
        board.digital[8].write(0)
        board.digital[9].write(0)
    elif st == 1:  # влево
        board.digital[2].write(1)
        board.digital[3].write(0)
        board.digital[4].write(0)
        board.digital[5].write(0)
        board.digital[8].write(0)
        board.digital[9].write(0)
    elif st == 2:  # вправо
        board.digital[2].write(0)
        board.digital[3].write(1)
        board.digital[4].write(0)
        board.digital[5].write(0)
        board.digital[8].write(0)
        board.digital[9].write(0)
    elif st == 3:  # назад
        board.digital[2].write(0)
        board.digital[3].write(0)
        board.digital[4].write(0)
        board.digital[5].write(1)
        board.digital[8].write(0)
        board.digital[9].write(0)
    elif st == 4:  # вперёд
        board.digital[2].write(0)
        board.digital[3].write(0)
        board.digital[4].write(1)
        board.digital[5].write(0)
        board.digital[8].write(0)
        board.digital[9].write(0)
    elif st == 5:  # вниз
        board.digital[2].write(0)
        board.digital[3].write(0)
        board.digital[4].write(0)
        board.digital[5].write(0)
        board.digital[8].write(0)
        board.digital[9].write(1)
    elif st == 6:  # вверх
        board.digital[2].write(0)
        board.digital[3].write(0)
        board.digital[4].write(0)
        board.digital[5].write(0)
        board.digital[8].write(1)
        board.digital[9].write(0)


# вывод информации о состоянии системы. Расстояние по всем осям, текущее действие робота, жест, углы
def text(cam, deltaX, deltaY, deltaZ, depth, status, gest, AL, AR):
    cv2.rectangle(cam, (0, 0), (640, 150), (255, 255, 255), -1)
    cv2.putText(cam, f"Depth: {int(depth)} cm", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Delta X: {int(deltaX)} cm", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Delta Y: {int(deltaY)} cm", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Delta Z: {int(deltaZ)} cm", (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Gest: {gest}", (0, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Dist Between Cam: {distanceBetweenCameras*10} mm", (210, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                1)
    cv2.putText(cam, f"FPS: 30 fps", (210, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Angle Left: {int(AL)}*", (210, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    cv2.putText(cam, f"Angle Right: {int(AR)}*", (210, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    if status == 0:
        cv2.putText(cam, f"Moving: Stop", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    elif status == 1:  # влево
        cv2.putText(cam, f"Moving: Right", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    elif status == 2:  # вправо
        cv2.putText(cam, f"Moving: Left", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    elif status == 3:  # назад
        cv2.putText(cam, f"Moving: Forward", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    elif status == 4:  # вперёд
        cv2.putText(cam, f"Moving: Back", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    elif status == 5:  # вниз
        cv2.putText(cam, f"Moving: Down", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    elif status == 6:  # вверх
        cv2.putText(cam, f"Moving: Up", (210, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    return cam


# распознавание поворота и захвата
def gestures(X, cord):
    gest = "No Gesture"
    if abs(X[4] - X[8]) < 4 and abs(X[4] - X[12]) < 4 and abs(X[4] - X[16]) < 4 and abs(X[4] - X[20]) < 4:
        gest = "Catch"

    AB = abs(cord[8][0] - cord[0][0])
    BC = abs(cord[8][1] - cord[0][1])
    if BC != 0 and abs(np.tan(AB / BC)) > 1:
        gest = "Turn"
    return gest


# нанесение на изображение виртуального манипулятора для наглядности
def virtualManipulator(cam, st):
    X_Rect, Y_Rect = cam.shape[1], cam.shape[0]
    point1 = np.array([[100, Y_Rect - 10], [70, Y_Rect - 10], [40, Y_Rect - 80], [70, Y_Rect - 80]])
    point2 = np.array([[X_Rect - 40, Y_Rect - 50], [X_Rect - 70, Y_Rect - 50], [X_Rect - 100, Y_Rect - 120],
                       [X_Rect - 70, Y_Rect - 120]])
    cv2.polylines(cam, [point1], 1, black, bold)
    cv2.polylines(cam, [point2], 1, black, bold)

    X1 = 85
    Y1 = Y_Rect - 45
    X2 = X_Rect - 85
    Y2 = Y_Rect - 85

    k = (Y1 - Y2) / (X2 - X1)

    X3 = int(X1 + (X2 - X1) / 2)
    Y3 = int(Y1 - X3 * k) + 15
    X4 = X3
    # Y4 = int(Y_Rect/2)
    Y4 = Y3 - 180

    X5 = X3
    Y5 = int(Y3 - (Y3 - Y4) / 2)
    X6 = X5 + 70
    Y6 = Y5 + 45
    X7 = X5 - 70
    Y7 = Y5 - 45

    cv2.line(cam, (X1, Y1), (X2, Y2), black, bold)
    cv2.line(cam, (X3, Y3), (X4, Y4), black, bold)
    cv2.line(cam, (X6, Y6), (X7, Y7), black, bold)

    if st == 1 or st == 2:
        cv2.line(cam, (X1, Y1), (X2, Y2), green, bold)
    elif st == 3 or st == 4:
        cv2.line(cam, (X6, Y6), (X7, Y7), green, bold)
    elif st == 5 or st == 6:
        cv2.line(cam, (X3, Y3), (X4, Y4), green, bold)
    return cam

while True:
    let1, leftCam = leftCamera.read()  # читаем изображение
    let2, rightCam = rightCamera.read()  # читаем изображение
    rightCam = cv2.flip(rightCam, 1)
    leftCam = cv2.flip(leftCam, 1)

    leftCam, cordFingerLeftCam = neuralNetwork(leftCam)
    rightCam, cordFingerRightCam = neuralNetwork(rightCam)

    if cordFingerLeftCam[0] is not None and cordFingerRightCam[0] is not None:
        if unlockingRightCam is False or unlockingLeftCam is False:
            leftCam, unlockingLeftCam = activate(leftCam, cordFingerLeftCam[8])
            rightCam, unlockingRightCam = activate(rightCam, cordFingerRightCam[8])
        else:
            leftCam = lines(leftCam)
            rightCam = lines(rightCam)

            # cмещение относительно камер в пикселях по горизонтальной и вертикальной осей. H - горизонталь, V - вертикаль
            leftHorizontalMiro = miroLH(cordFingerLeftCam, 640)
            rightHorizontalMiro = miroRH(cordFingerRightCam, 640)
            leftVerticalMiro = miroLV(cordFingerLeftCam, 480)

            # cмещение относительно камер в градусах по горизонтальной и вертикальной осей
            leftHorizontalAngle = angle(76, 640, leftHorizontalMiro)
            rightHorizontalAngle = angle(76, 640, rightHorizontalMiro)
            leftVerticalAngle = angle(56, 480, leftVerticalMiro)

            # определение расстояния до точек
            depth = longH(leftHorizontalAngle, rightHorizontalAngle, distanceBetweenCameras)

            # определение смещения в сантиметрах, вместо гадусов и пикселей. Для глубины находим модуль (70 -
            # расстояние) для проверки на пороговое значение
            longX = camX(depth, leftHorizontalAngle)
            longY = camY(depth)
            longZ = camZ(depth, leftVerticalAngle)

            # определние жеста
            gest = gestures(longX, cordFingerLeftCam)
            # определение текущей команды для движения
            st = definitionMovement(longX[8], longY[8], longZ[8], gest)

            # рисование интерфейса
            leftCam = paintGryd(leftCam, st, gest)
            rightCam = text(rightCam, longX[8], longY[8], longZ[8], depth[8], st, gest, leftHorizontalAngle[8],
                            rightHorizontalAngle[8])
            rightCam = virtualManipulator(rightCam, st)
            # отправка команды на ардуино
            go(st)

    else:
        unlockingLeftCam = False
        unlockingRightCam = False
        go(0)

    img = np.concatenate((leftCam, rightCam), 1)
    cv2.imshow("1", img)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
