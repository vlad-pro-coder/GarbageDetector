import cv2
import tensorflow as tf
import numpy as np

# Example: preprocess and run inference

class YOLOMultiHeadModel(tf.keras.Model):
    def __init__(self, backbone_model, loss_fn,**kwargs):
        super(YOLOMultiHeadModel, self).__init__(**kwargs)
        self.backbone = backbone_model
        self.loss_fn = loss_fn
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        grads_and_vars = []
        for grad, var in zip(gradients, self.trainable_variables):
            if grad is None:
                tf.print(f"No gradients for variable: {var.name}")
            else:
                grads_and_vars.append((grad, var))
        self.optimizer.apply_gradients(grads_and_vars)
        self.loss_tracker.update_state(loss)
        tf.print("\n")
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        images, labels = data
        predictions = self(images, training=False)
        loss = self.loss_fn(labels, predictions)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

NUM_CLASSES = 26
NUM_ANCHORS = 3
constant_anchors = {
    #"52":[[10,13], [16,30], [33,23]],
    26:[[7.90196078,5.824],[ 17.46078431,16.19298246],[ 43.03448105,39.25490196]],
    13:[[ 95.50221204,73.70000839],[145.99999905,161.99999809],[255.31603053,238.99232006]],
}

ANCHORS = np.array([[7.90196078,5.824],[ 17.46078431,16.19298246],[ 43.03448105,39.25490196],[ 95.50221204,73.70000839],[145.99999905,161.99999809],[255.31603053,238.99232006]])

ANCHOR_INDECES = {
    26:[0,1,2],
    13:[3,4,5],
}

PHOTO_SIZE = 416

# Windows IP address (localhost usually works in WSL2)
url = "http://192.168.0.198:5000/video"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_predictions(pred, conf_thresh=0.64):
    batch_size, grid, _, anchors, channels = pred.shape
    pred = pred[0].numpy()
    boxes = []
    pixels_per_grid = 416 // grid
    for i in range(grid):
        for j in range(grid):
            for k in range(anchors):
                obj_score = sigmoid(pred[i, j, k, 4])
                if obj_score > conf_thresh:
                    bx, by, bw, bh = pred[i, j, k, 0:4]
                    class_probs = sigmoid(pred[i, j, k, 5:])
                    class_id = np.argmax(class_probs)
                    score = obj_score * class_probs[class_id]
                    boxes.append([bx + pixels_per_grid * i, by + pixels_per_grid * j, bw, bh, score, class_id])
    return boxes

def preprocess_image(image, img_size):
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0  # normalize
    return tf.expand_dims(image, axis=0)
img_size = 416  # or whatever your model uses

def normalize_yolo_output(y_pred):
    """
    y_pred: (batch, S, S, 3, 5 + C)
    anchors: (3, 2) — anchor box sizes (width, height) in pixels
    grid_size: int — S = 13 or 26 typically

    Returns:
        Tensor of shape (batch, S, S, 3, 5 + C) with:
        - objectness: sigmoid
        - class scores: softmax
    """
    grid_size = y_pred.shape[1]
    anchors = tf.convert_to_tensor(constant_anchors[grid_size], dtype=tf.float32)  # (3, 2)

    # Ensure anchor shape is (1, 1, 1, 3, 2) to broadcast
    anchors = tf.reshape(anchors, (1, 1, 1, NUM_ANCHORS, 2))

    # Slice components
    tx = tf.sigmoid(y_pred[..., 0]) * (PHOTO_SIZE // grid_size)
    ty = tf.sigmoid(y_pred[..., 1]) * (PHOTO_SIZE // grid_size)
    tw = tf.exp(y_pred[..., 2]) * anchors[..., 0]
    th = tf.exp(y_pred[..., 3]) * anchors[..., 1]
    obj = tf.sigmoid(y_pred[..., 4])
    cls = tf.nn.softmax(y_pred[..., 5:], axis=-1)

    # Stack normalized values
    bbox = tf.stack([tx, ty, tw, th, obj], axis=-1)  # shape: (batch, S, S, 3, 5)
    out = tf.concat([bbox, cls], axis=-1)           # shape: (batch, S, S, 3, 5 + C)

    return out

def convert_box(cx, cy, w, h, img_w, img_h):
    x = int((cx - w / 2))
    y = int((cy - h / 2))
    w = int(w)
    h = int(h)
    return x, y, w, h

def draw_boxes_on_frame(image, boxes, class_names=None):
    """
    Draws bounding boxes on a 416x416x3 frame.
    
    Args:
        image: np.array of shape (416, 416, 3), dtype=np.uint8
        boxes: list/array of boxes, each box = [cx, cy, w, h, score, class_id]
        class_names: optional list of class names
    Returns:
        image with boxes drawn
    """
    frame = image.copy()

    for box in boxes:
        cx, cy, w, h, score, class_id = box
        x, y, bw, bh = convert_box(cx, cy, w, h, frame.shape[1], frame.shape[0])
        
        # Draw rectangle
        cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)

        # Optional: label
        if class_names:
            label = f"{class_names[int(class_id)]}: {score:.2f}"
            cv2.putText(frame, label, (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

cap = cv2.VideoCapture(url)
model = tf.keras.models.load_model('./GarbageDetector101.h5', custom_objects={"YOLOMultiHeadModel": YOLOMultiHeadModel})

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("⚠️ Failed to grab frame")
        continue

    image_tensor = preprocess_image(frame,img_size)
    
    preds = model(image_tensor)

    preds = [
    normalize_yolo_output(tf.reshape(preds[0],(1, 13, 13, NUM_ANCHORS, 5 + NUM_CLASSES))),
    normalize_yolo_output(tf.reshape(preds[1],(1, 26, 26, NUM_ANCHORS, 5 + NUM_CLASSES)))
    ]
    grid_size_13 = preds[0]
    grid_size_26 = preds[1]

    filtered_grid_size_13 = decode_predictions(grid_size_13)
    filtered_grid_size_26 = decode_predictions(grid_size_26)

    new_frame = draw_boxes_on_frame(tf.squeeze(image_tensor, axis=0),filtered_grid_size_13)
    
    if not ret:
        continue
    # Now frame is a normal OpenCV image, ready for AI processing
    # Example: show frame (optional)
    cv2.imshow("Stream", new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()