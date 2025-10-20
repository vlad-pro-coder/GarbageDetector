import cv2
import numpy as np
from openvino import Core,properties

# ----------------------------
# Load OpenVINO model
# ----------------------------
CONF_THRESH = 0.35
NMS_THRESH = 0.5
CLASSES = [
    'Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton',
    'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper',
    'Plastic bag - wrapper', 'Plastic container', 'Pop tab', 'Straw',
    'Styrofoam piece', 'Unlabeled litter'
]

ie = Core()
model_path = "best.xml"
weights_path = "best.bin"
compiled_model = ie.compile_model(model_path, "CPU",config={properties.inference_num_threads(): 24})


# Get input and output layer info
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print("Input shape:", input_layer.shape)
print("Output shape:", output_layer.shape)

# ----------------------------
# Preprocess image
# ----------------------------
def format_to_square(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)

    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    # NCNN/OpenVINO expect [N,C,H,W]

    canvas_uint8 = (canvas * 255.0).astype(np.uint8)
    cv2.imwrite("canvas.jpg", canvas_uint8)
    #print(canvas)

    blob = np.expand_dims(np.transpose(canvas, (2,0,1)), axis=0)
    return blob.astype(np.float32), scale, pad_x, pad_y

# Load image
img = cv2.imread("../WhatsApp Image 2025-09-14 at 22.50.07.jpeg").astype(np.float32) / 255.0
orig_img = cv2.imread("../WhatsApp Image 2025-09-14 at 22.50.07.jpeg")
input_blob, scale, pad_x, pad_y = format_to_square(img)

# ----------------------------
# Run inference
# ----------------------------
result = compiled_model([input_blob])[output_layer]
print("Raw output shape:", result.shape)

# ----------------------------
# Postprocess
# ----------------------------
def process_yolo_output(out, scale, pad_x, pad_y, orig_shape, score_threshold=0.45, nms_threshold=0.5):
    """
    out: [num_classes+4, num_detections] like YOLOv11 NCNN output
    Returns: list of [x, y, w, h, score, class_id]
    """ 
    boxes, confidences, class_ids = [], [], []
    out = np.squeeze(out, axis=0)  
    num_dets = out.shape[1]
    for i in range(num_dets):
        cx, cy, w, h = out[0:4, i]
        #print(cx, cy, w, h)
        class_scores = out[4:, i]
        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])
        if conf < score_threshold:
            continue
        left   = int((cx - 0.5*w - pad_x) / scale)
        top    = int((cy - 0.5*h - pad_y) / scale)
        width  = int(w / scale)
        height = int(h / scale)

        boxes.append([left, top, width, height])
        confidences.append(conf)
        class_ids.append(cls_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    results = []
    for i in indices:
        x, y, w, h = boxes[i]
        results.append([x, y, w, h, confidences[i], class_ids[i]])
    return results

predictions = process_yolo_output(result, scale, pad_x, pad_y, img.shape)
print(predictions)

def draw_detections(img, detections):
    for (x, y, w, h, score, cls_id) in detections:
        label = f"{CLASSES[cls_id]} {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

drawn_img = draw_detections(orig_img,predictions)

cv2.imwrite("result.jpg", drawn_img)
