import ncnn
import numpy as np
import cv2

# -------------------------
# Config
# -------------------------
INPUT_SIZE = 640
CONF_THRESH = 0.45
NMS_THRESH = 0.50
CLASSES = [
    'Aluminium foil', 'Bottle cap', 'Bottle', 'Broken glass', 'Can', 'Carton',
    'Cigarette', 'Cup', 'Lid', 'Other litter', 'Other plastic', 'Paper',
    'Plastic bag - wrapper', 'Plastic container', 'Pop tab', 'Straw',
    'Styrofoam piece', 'Unlabeled litter'
]

# -------------------------
# Model loader
# -------------------------
net = ncnn.Net()
net.load_param("./model.ncnn.param") 
net.load_model("./model.ncnn.bin")

# -------------------------
# Preprocess: letterbox resize
# -------------------------
def format_to_square(img, new_shape=(INPUT_SIZE, INPUT_SIZE)):

    h, w = img.shape[:2]
    scale = min(new_shape[0] / w, new_shape[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((new_shape[1], new_shape[0], 3), dtype=np.float32)
    pad_x, pad_y = (new_shape[0] - new_w) // 2, (new_shape[1] - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # NCNN expects [C,H,W]
    #cv2.imwrite("test.jpg", canvas * 255.0)
    blob = np.expand_dims(np.transpose(canvas, (2, 0, 1)), axis=0)
    
    return blob, scale, pad_x, pad_y

# -------------------------
# Inference
# -------------------------
def run_inference(img):
    blob, scale, pad_x, pad_y = format_to_square(img)

    print(blob.shape)

    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(np.array(blob[0],copy=True)))
        _, out = ex.extract("out0")
        #print(np.array(out).shape)
        out = np.array(out).copy().T  # copy to avoid stale buffer

    # Each row = [cx, cy, w, h, obj_conf, class_probs...]
    return out, scale, pad_x, pad_y

# -------------------------
# Postprocess
# -------------------------
def process_results(out, scale, pad_x, pad_y, orig_shape):
    h0, w0 = orig_shape[:2]

    boxes, scores, class_ids = [], [], []

    for det in out:
        cx, cy, w, h = det[0:4]
        class_scores = det[4:]   # directly class probabilities

        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])   # no obj_conf in YOLOv11

        if conf < CONF_THRESH:
            continue

        # Box in padded image coords
        left   = int((cx - 0.5 * w))
        top    = int((cy - 0.5 * h))
        width  = int(w)
        height = int(h)

        boxes.append([left, top, width, height])
        scores.append(conf)
        class_ids.append(cls_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    results = []
    for i in indices:
        x, y, w, h = boxes[i]
        results.append((x, y, w, h, scores[i], class_ids[i]))

    return results


# -------------------------
# Draw results
# -------------------------
def draw_detections(img, detections):
    for (x, y, w, h, score, cls_id) in detections:
        label = f"{CLASSES[cls_id]} {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

# -------------------------
# Run test
# -------------------------
if __name__ == "__main__":
    img = cv2.imread("../capture.jpg").astype(np.float32) / 255.0
    out, scale, pad_x, pad_y = run_inference(img)
    detections = process_results(out, scale, pad_x, pad_y, img.shape)

    print(detections)
    img_out = draw_detections((img*255).astype(np.uint8), detections)
    cv2.imwrite("predictions.jpg", img_out)
    print("Saved predictions.jpg")
