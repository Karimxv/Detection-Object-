import cv2
import numpy as np

# تحميل ملف التكوين والأوزان لنموذج YOLO
net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')

# قراءة قائمة الأصناف (ملف الأصناف يجب أن يكون متاحًا)
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# قراءة الصورة
img = cv2.imread('image.jpg')

# إعداد الصورة كبيانات الإدخال المطلوبة لنموذج YOLO
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# إدخال الصورة إلى النموذج والحصول على الإخراج
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# تحليل الإخراج وتحديد الكائنات
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # التأكد من الثقة
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # إحداثيات الركن العلوي الأيمن والركن السفلي الأيسر
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# الرسم على الصورة
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

# عرض الصورة
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
