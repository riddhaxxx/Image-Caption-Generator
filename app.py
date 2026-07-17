import streamlit as st
import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from collections import Counter
from PIL import Image
import tempfile
import matplotlib
matplotlib.use("Agg")

# -------------------------------
# TensorFlow / Keras Imports
# -------------------------------
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)

from tensorflow.keras.preprocessing.sequence import (
    pad_sequences
)

from tensorflow.keras.models import load_model

# -------------------------------
# YOLO
# -------------------------------
from ultralytics import YOLO

# -------------------------------
# Streamlit Page
# -------------------------------

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Caption Generator")

st.write(
    """
Upload an image and generate captions using

• YOLOv8
• ResNet50
• Hybrid LSTM
• Beam Search
"""
)

# -------------------------------
# Load Models
# -------------------------------

@st.cache_resource
def load_all_models():

    yolo_model = YOLO("yolov8n.pt")

    resnet_model = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    with open("tokenizer.pkl","rb") as f:
        tokenizer = pickle.load(f)

    caption_model = load_model(
        "caption_model.keras"
    )

    MAX_LENGTH = 34

    return (
        yolo_model,
        resnet_model,
        tokenizer,
        caption_model,
        MAX_LENGTH
    )

with st.spinner("Loading models..."):

    (
        yolo_model,
        resnet_model,
        tokenizer,
        lstm_model,
        MAX_LENGTH
    ) = load_all_models()

st.success("Models Loaded Successfully!")

# -------------------------------
# Feature Extraction
# -------------------------------

def extract_image_features(img_path):

    yolo_results = yolo_model(
        img_path,
        verbose=False
    )[0]

    boxes = yolo_results.boxes

    num_objs = len(boxes)

    yolo_feats = []

    detected_classes = []

    names = yolo_model.names

    if num_objs > 0:

        conf_scores = boxes.conf.cpu().numpy()

        sorted_indices = np.argsort(
            conf_scores
        )[::-1]

        for i in range(min(15,num_objs)):

            idx = sorted_indices[i]

            b = boxes[idx]

            class_id = float(
                b.cls[0].cpu().numpy()
            )

            conf = float(
                b.conf[0].cpu().numpy()
            )

            if conf > 0.40:

                detected_classes.append(
                    names[int(class_id)]
                )

            cx,cy,w,h = (
                b.xywhn[0]
                .cpu()
                .numpy()
                .tolist()
            )

            area = w*h

            yolo_feats.extend([
                class_id,
                conf,
                cx,
                cy,
                w,
                h,
                area,
                cx,
                cy
            ])

    padding_needed = 135-len(yolo_feats)

    yolo_feats.extend(
        [0.0]*padding_needed
    )

    yolo_feats.append(
        float(num_objs)
    )

    yolo_vector = np.array(
        yolo_feats
    )

    img = load_img(
        img_path,
        target_size=(224,224)
    )

    img_array = img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    resnet_vector = resnet_model.predict(
        img_array,
        verbose=0
    ).flatten()

    combined_vector = np.concatenate(
        (
            resnet_vector,
            yolo_vector
        )
    )

    return (
        np.array([combined_vector]),
        yolo_results,
        detected_classes
    )
  # -------------------------------------------------------
# Convert Integer to Word
# -------------------------------------------------------

def int_to_word(integer, tokenizer):

    for word, index in tokenizer.word_index.items():

        if index == integer:
            return word

    return None


# -------------------------------------------------------
# Beam Search Caption Generator
# -------------------------------------------------------

def generate_caption_beam_search(
        model,
        tokenizer,
        image_feature,
        max_length,
        detected_classes,
        beam_width=5,
        alpha=0.8
):

    start_word = "<start>"

    beam = [([start_word], 0.0)]

    img_tensor = tf.convert_to_tensor(
        image_feature
    )

    stop_words = {
        "a","the","and","is","in",
        "on","of","with","at",
        "to","by","an","are"
    }

    class_counts = Counter(
        detected_classes
    )

    for _ in range(max_length):

        candidates = []

        for seq, score in beam:

            if seq[-1] == "end":

                candidates.append(
                    (seq, score)
                )

                continue

            seq_str = " ".join(seq)

            encoded_seq = tokenizer.texts_to_sequences(
                [seq_str]
            )[0]

            padded_seq = pad_sequences(
                [encoded_seq],
                maxlen=max_length,
                padding="post"
            )

            seq_tensor = tf.convert_to_tensor(
                padded_seq
            )

            yhat = model.predict_on_batch(
                [
                    img_tensor,
                    seq_tensor
                ]
            )[0]

            current_max_prob = np.max(yhat)

            yolo_seen_broad = set()

            for class_name, count in class_counts.items():

                boost_words = [class_name]

                if class_name == "person":

                    if count > 1:

                        boost_words.extend([
                            "women",
                            "men",
                            "people",
                            "group",
                            "girls",
                            "friends"
                        ])

                    else:

                        boost_words.extend([
                            "man",
                            "woman",
                            "boy",
                            "girl",
                            "runner"
                        ])

                elif class_name == "dog":

                    if count > 1:

                        boost_words.extend([
                            "dogs",
                            "puppies",
                            "pack"
                        ])

                    else:

                        boost_words.extend([
                            "puppy",
                            "hound"
                        ])

                elif class_name == "cat":

                    if count > 1:

                        boost_words.extend([
                            "cats",
                            "kittens"
                        ])

                    else:

                        boost_words.extend([
                            "kitten",
                            "feline"
                        ])

                elif class_name == "car":

                    if count > 1:

                        boost_words.extend([
                            "cars",
                            "vehicles"
                        ])

                    else:

                        boost_words.extend([
                            "vehicle",
                            "automobile"
                        ])

                elif class_name == "handbag":

                    boost_words.extend([
                        "bag",
                        "purse"
                    ])

                yolo_seen_broad.update(
                    boost_words
                )

                if not any(
                    w in seq
                    for w in boost_words
                ):

                    for w in boost_words:

                        if w in tokenizer.word_index:

                            word_idx = tokenizer.word_index[w]

                            yhat[word_idx] += (
                                current_max_prob * 0.35
                            )

            common_biases = {
                "dog",
                "dogs",
                "man",
                "woman",
                "boy",
                "girl",
                "person",
                "people",
                "child",
                "children"
            }

            hallucination_risks = (
                common_biases
                - yolo_seen_broad
            )

            for risk_word in hallucination_risks:

                if risk_word in tokenizer.word_index:

                    word_idx = tokenizer.word_index[
                        risk_word
                    ]

                    yhat[word_idx] *= 0.001

            for word in set(seq):

                if (
                    word not in stop_words
                    and word in tokenizer.word_index
                ):

                    word_idx = tokenizer.word_index[word]

                    yhat[word_idx] *= 0.001

            if len(seq) > 0:

                last_word = seq[-1]

                if last_word in tokenizer.word_index:

                    yhat[
                        tokenizer.word_index[last_word]
                    ] *= 0.001

            yhat = yhat / (
                np.sum(yhat) + 1e-10
            )

            top_indices = np.argsort(
                yhat
            )[-beam_width:]

            for idx in top_indices:

                word = int_to_word(
                    idx,
                    tokenizer
                )

                if word is None:
                    continue

                prob = yhat[idx]

                new_score = (
                    score
                    - np.log(prob + 1e-10)
                )

                new_seq = seq + [word]

                candidates.append(
                    (
                        new_seq,
                        new_score
                    )
                )

        def score_with_length_penalty(item):

            seq, current_score = item

            L = len(seq) - 1

            penalty = (
                L ** alpha
            ) if L > 0 else 1.0

            return current_score / penalty

        beam = sorted(
            candidates,
            key=score_with_length_penalty
        )[:beam_width]

        if all(
            seq[-1] == "end"
            for seq, _ in beam
        ):
            break

    best_seq = beam[0][0]

    if best_seq[0] == "<start>":
        best_seq = best_seq[1:]

    if len(best_seq) > 0 and best_seq[-1] == "end":
        best_seq = best_seq[:-1]

    return " ".join(best_seq).strip()
  # -------------------------------------------------------
# Streamlit Interface
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🚀 Generate Caption"):

        with st.spinner("Generating Caption..."):

            # Save uploaded image temporarily
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            )

            temp_file.write(uploaded_file.read())
            temp_file.close()

            try:

                # Feature Extraction
                (
                    img_feature,
                    yolo_results,
                    detected_classes
                ) = extract_image_features(
                    temp_file.name
                )

                # Caption Generation
                caption = generate_caption_beam_search(
                    lstm_model,
                    tokenizer,
                    img_feature,
                    MAX_LENGTH,
                    detected_classes,
                    beam_width=5,
                    alpha=0.8
                )

                st.success("Caption Generated Successfully!")

                st.subheader("📝 Generated Caption")

                st.write(caption.capitalize() + ".")

                st.subheader("🎯 Objects Detected")

                if len(detected_classes) == 0:
                    st.write("No objects detected.")

                else:
                    st.write(", ".join(detected_classes))

                # Draw YOLO detections
                plotted_img = yolo_results.plot()

                plotted_img = cv2.cvtColor(
                    plotted_img,
                    cv2.COLOR_BGR2RGB
                )

                st.subheader("📸 Detection Result")

                st.image(
                    plotted_img,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Error : {e}"
                )

            finally:

                if os.path.exists(
                    temp_file.name
                ):
                    os.remove(
                        temp_file.name
                    )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------

st.markdown("---")

st.markdown(
    """
Developed using

- YOLOv8
- ResNet50
- TensorFlow LSTM
- Beam Search
- Streamlit
"""
)
