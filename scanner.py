import cv2
import numpy as np
import imutils

def order_points(pts):
   rect = np.zeros((4, 2), dtype = "float32")
   s = pts.sum(axis = 1)
   rect[0] = pts[np.argmin(s)]
   rect[2] = pts[np.argmax(s)]
   diff = np.diff(pts, axis = 1)
   rect[1] = pts[np.argmin(diff)]
   rect[3] = pts[np.argmax(diff)]
   return rect

def perspective_transform(image, pts):
   rect = order_points(pts)
   (tl, tr, br, bl) = rect
   widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
   widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
   maxWidth = max(int(widthA), int(widthB))
   heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
   heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
   maxHeight = max(int(heightA), int(heightB))

   dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")

   transform_matrix = cv2.getPerspectiveTransform(rect, dst)
   warped = cv2.warpPerspective(image, transform_matrix, (maxWidth, maxHeight))
   return warped

def scan_document(image_path):
    img = cv2.imread(image_path)
    copy = img.copy()
    ratio = img.shape[0] / 500.0
    img_resize = imutils.resize(img, height=500)

    gray_image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    edged_img = cv2.Canny(blurred_image, 75, 200)

    cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc = approx
            break

    warped_image = perspective_transform(copy, doc.reshape(4, 2) * ratio)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    return warped_image