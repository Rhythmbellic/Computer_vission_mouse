import cv2
import numpy as np
import pyautogui

cam = cv2.VideoCapture(0)

lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Initialize variables to track clicks
left_clicked = False
right_clicked = False

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Smoothen the image
    image_smooth = cv2.GaussianBlur(frame, (7, 7), 0)

    # Define ROI
    mask = np.zeros_like(frame)
    mask[50:350, 50:350] = [255, 255, 255]
    image_roi = cv2.bitwise_and(image_smooth, mask)

    # Draw a yellow grid
    for i in range(4):
        cv2.line(frame, (50 + i * 100, 50), (50 + i * 100, 350), (0, 255, 255), 2)  # Vertical lines (thicker)
        cv2.line(frame, (50, 50 + i * 100), (350, 50 + i * 100), (0, 255, 255), 2)  # Horizontal lines (thicker)

    # Threshold the image for blue color (cursor)
    image_hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    image_threshold = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # Finding contours for blue
    contour_blue, hierarchy_blue = cv2.findContours(image_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Index of the largest blue contour
    if len(contour_blue) != 0:
        area = [cv2.contourArea(c) for c in contour_blue]
        max_index = np.argmax(area)
        cnt = contour_blue[max_index]

        # Pointer
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Draw a blue circle where blue color is detected
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)

            # Cursor motion
            if cx < 150:
                dist_x = -20
            elif cx > 250:
                dist_x = 20
            else:
                dist_x = 0

            if cy < 150:
                dist_y = -20
            elif cy > 250:
                dist_y = 20
            else:
                dist_y = 0
            pyautogui.moveRel(dist_x, dist_y, duration=0.25)

    # Check for right-click (green color)
    image_threshold_green = cv2.inRange(image_hsv, lower_green, upper_green)
    contour_green, hierarchy_green = cv2.findContours(image_threshold_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contour_green) != 0:
        if not right_clicked:
            right_clicked = True
            pyautogui.rightClick()
            right_clicked = True
    else:
        right_clicked = False
            

    # Check for left-click (red color)
    image_threshold_red = cv2.inRange(image_hsv, lower_red, upper_red)
    contour_red, hierarchy_red = cv2.findContours(image_threshold_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contour_red) != 0:
        if not left_clicked:
            pyautogui.click()
            left_clicked = True
    else:
        left_clicked = False

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1000)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
