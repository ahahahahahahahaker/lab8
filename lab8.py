# Вариант 3
import cv2
import numpy as np

# 1 задание
apple = cv2.imread('variant-3.jpeg')
apple = cv2.resize(apple, dsize=(1000, 600), fx=100, fy=100)
hsv_image = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2 задание


def task_2():
    video = cv2.VideoCapture(0)
    image = cv2.imread('ref-point.jpg', 0)
    imagesize = image.shape[:2]
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = 0.5
        result = cv2.matchTemplate(gray, image, cv2.TM_CCOEFF_NORMED)
        locations = []
        for y, x in zip(*np.where(result >= threshold)):
            locations.append((x, y))
        for loc in locations:
            cv2.rectangle(frame, (loc[0], loc[1]), (loc[0]+imagesize[0], loc[1]+imagesize[1]), (0, 255, 255), 3)

        framesize = frame.shape
        rectsize = (200, 200)
        cords = (framesize[1]//2 - rectsize[0]//2, framesize[0]//2 - rectsize[1]//2)
        cv2.rectangle(frame, (cords[0], cords[1]), (cords[0] + rectsize[0], cords[1]+rectsize[1]),
                      (255, 255, 255), 3)
        for loc in locations:
            if (cords[0] < loc[0]+imagesize[0]//2 < cords[0] + rectsize[0]) \
                    and (cords[1] < loc[1]+imagesize[1]//2 < cords[1] + rectsize[1]):
                cv2.rectangle(frame, (cords[0], cords[1]), (cords[0] + rectsize[0],
                                                            cords[1]+rectsize[1]), (255, 0, 0), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video.release()


#  Доп задание
def task_4():
    video = cv2.VideoCapture(0)
    image = cv2.imread('ref-point.jpg', 0)
    imagesize = image.shape[:2]
    while True:
        ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = 0.5
        result = cv2.matchTemplate(gray, image, cv2.TM_CCOEFF_NORMED)
        locations = []
        for y, x in zip(*np.where(result >= threshold)):
            locations.append((x, y))
        for loc in locations:
            cv2.rectangle(frame, (loc[0], loc[1]), (loc[0]+imagesize[0], loc[1]+imagesize[1]), (0, 255, 255), 3)
        #
        #
        #
        flyimage = cv2.imread('fly64.png')
        flyshape = flyimage.shape[:2]
        for loc in locations:
            x, y = loc
            x += imagesize[1]//2-flyshape[1]//2
            y += imagesize[0]//2-flyshape[0]//2
            try:
                frame[y:y + flyshape[0], x:x + flyshape[1]] = flyimage[:, :]
            except:
                print('Bzzz -_-')
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()


if __name__ == '__main__':
    task_2()
#    task_4()
