import cv2

def putBorderedText(image, string, pos, font, font_scale=1, size=4):
    cv2.putText(image, string, pos, font, font_scale, (0, 0, 0), size+8)
    cv2.putText(image, string, pos, font, font_scale, (255, 255, 255), size)