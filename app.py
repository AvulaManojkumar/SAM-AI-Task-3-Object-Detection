import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort

st.set_page_config(page_title="Object Detection", page_icon="🎯")

st.title("🎯 Object Detection")
st.write("Upload an image to detect common objects.")

# Load class names
with open("coco.names", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO ONNX model
session = ort.InferenceSession(
    "yolo11n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = cv2.imdecode(
        np.frombuffer(uploaded_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    st.subheader("Original Image")
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    if st.button("🔍 Detect Objects"):

        height, width = image.shape[:2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to YOLO input size
        resized = cv2.resize(rgb, (640, 640))

        # Normalize
        input_image = resized.astype(np.float32) / 255.0

        # HWC -> CHW
        input_image = np.transpose(input_image, (2, 0, 1))

        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)

        # Run model
        output = session.run(
            None,
            {input_name: input_image}
        )[0]

        raw = output[0]

        # Handle YOLO output format
        if raw.shape[0] < raw.shape[1]:
            predictions = raw.T
        else:
            predictions = raw

        boxes = []
        scores = []
        class_ids = []

        for prediction in predictions:

            class_scores = prediction[4:]

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < 0.35:
                continue

            x, y, w, h = prediction[:4]

            x1 = int((x - w / 2) * width / 640)
            y1 = int((y - h / 2) * height / 640)
            x2 = int((x + w / 2) * width / 640)
            y2 = int((y + h / 2) * height / 640)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        # Non-Maximum Suppression
        indexes = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            0.35,
            0.45
        )

        detected_objects = []

        if len(indexes) > 0:

            for i in np.array(indexes).flatten():

                x, y, w, h = boxes[i]

                class_id = class_ids[i]
                confidence = scores[i]

                label = classes[class_id]

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                text = f"{label}: {confidence:.2f}"

                cv2.putText(
                    image,
                    text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                detected_objects.append(
                    f"{label} — {confidence * 100:.1f}%"
                )

        st.subheader("Detected Objects")

        if detected_objects:
            for obj in detected_objects:
                st.success(obj)
        else:
            st.warning("No objects detected.")

        st.subheader("Detection Result")

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )