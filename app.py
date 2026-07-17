import os
import pickle
import numpy as np
import tempfile
from collections import Counter

import streamlit as st
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from ultralytics import YOLO

# -------------------------------------------------
# NOTE on filenames:
# Your repo currently has "caption_model (1).keras" and "tokenizer (1).pkl"
# (with a space + "(1)"). Rename them in the repo to the names below
# (no spaces, no parentheses) -- spaces in filenames cause problems with
# some tools/CLIs and it's best practice to avoid them.
# -------------------------------------------------
MODEL_PATH = "caption_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LENGTH = 34

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="centered")


@st.cache_resource(show_spinner="Loading models (first run can take a minute)...")
def load_models():
    yolo_model = YOLO("yolov8m.pt")  # auto-downloads on first run
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    lstm_model = load_model(MODEL_PATH)
    return yolo_model, resnet_model, tokenizer, lstm_model


def extract_image_features(img_path, yolo_model, resnet_model):
    yolo_results = yolo_model(img_path, verbose=False)[0]
    boxes = yolo_results.boxes
    num_objs = len(boxes)
    yolo_feats = []
    detected_classes = []
    names = yolo_model.names

    if num_objs > 0:
        conf_scores = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(conf_scores)[::-1]
        for i in range(min(15, num_objs)):
            idx = sorted_indices[i]
            b = boxes[idx]
            class_id = float(b.cls[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy())
            if conf > 0.40:
                detected_classes.append(names[int(class_id)])
            cx, cy, w, h = b.xywhn[0].cpu().numpy().tolist()
            area = w * h
            yolo_feats.extend([class_id, conf, cx, cy, w, h, area, cx, cy])

    padding_needed = 135 - len(yolo_feats)
    yolo_feats.extend([0.0] * padding_needed)
    yolo_feats.append(float(num_objs))
    yolo_vector = np.array(yolo_feats)

    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    resnet_vector = resnet_model.predict(img_array, verbose=0).flatten()

    combined_vector = np.concatenate((resnet_vector, yolo_vector))
    return np.array([combined_vector]), yolo_results, detected_classes


def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption_beam_search(model, tokenizer, image_feature, max_length,
                                  detected_classes, beam_width=5, alpha=0.8):
    start_word = "<start>"
    beam = [([start_word], 0.0)]
    img_tensor = tf.convert_to_tensor(image_feature)

    stop_words = {"a", "the", "and", "is", "in", "on", "of", "with",
                  "at", "to", "by", "an", "are"}
    class_counts = Counter(detected_classes)

    for _ in range(max_length):
        candidates = []
        for seq, score in beam:
            if seq[-1] == "end":
                candidates.append((seq, score))
                continue

            seq_str = " ".join(seq)
            encoded_seq = tokenizer.texts_to_sequences([seq_str])[0]
            padded_seq = pad_sequences([encoded_seq], maxlen=max_length, padding="post")
            seq_tensor = tf.convert_to_tensor(padded_seq)

            yhat = model.predict_on_batch([img_tensor, seq_tensor])[0]
            current_max_prob = np.max(yhat)
            yolo_seen_broad = set()

            for class_name, count in class_counts.items():
                boost_words = [class_name]
                if class_name == "person":
                    boost_words.extend(["women", "men", "people", "group", "girls", "friends"]
                                        if count > 1 else
                                        ["man", "woman", "boy", "girl", "runner"])
                elif class_name == "dog":
                    boost_words.extend(["dogs", "puppies", "pack"] if count > 1 else ["puppy", "hound"])
                elif class_name == "cat":
                    boost_words.extend(["cats", "kittens"] if count > 1 else ["kitten", "feline"])
                elif class_name == "car":
                    boost_words.extend(["cars", "vehicles"] if count > 1 else ["vehicle", "automobile"])
                elif class_name == "handbag":
                    boost_words.extend(["bag", "purse"])

                yolo_seen_broad.update(boost_words)
                if not any(w in seq for w in boost_words):
                    for w in boost_words:
                        if w in tokenizer.word_index:
                            yhat[tokenizer.word_index[w]] += current_max_prob * 0.35

            common_biases = {"dog", "dogs", "man", "woman", "boy", "girl",
                              "person", "people", "child", "children"}
            hallucination_risks = common_biases - yolo_seen_broad
            for risk_word in hallucination_risks:
                if risk_word in tokenizer.word_index:
                    yhat[tokenizer.word_index[risk_word]] *= 0.001

            for word in set(seq):
                if word not in stop_words and word in tokenizer.word_index:
                    yhat[tokenizer.word_index[word]] *= 0.001

            if seq:
                last_word = seq[-1]
                if last_word in tokenizer.word_index:
                    yhat[tokenizer.word_index[last_word]] *= 0.001

            yhat = yhat / (np.sum(yhat) + 1e-10)
            top_indices = np.argsort(yhat)[-beam_width:]

            for idx in top_indices:
                word = int_to_word(idx, tokenizer)
                if word is None:
                    continue
                prob = yhat[idx]
                new_score = score - np.log(prob + 1e-10)
                candidates.append((seq + [word], new_score))

        def score_with_length_penalty(item):
            seq, current_score = item
            L = len(seq) - 1
            penalty = (L ** alpha) if L > 0 else 1.0
            return current_score / penalty

        beam = sorted(candidates, key=score_with_length_penalty)[:beam_width]
        if all(seq[-1] == "end" for seq, _ in beam):
            break

    best_seq = beam[0][0]
    if best_seq and best_seq[0] == "<start>":
        best_seq = best_seq[1:]
    if best_seq and best_seq[-1] == "end":
        best_seq = best_seq[:-1]
    return " ".join(best_seq).strip()


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🖼️ AI Image Caption Generator")
st.caption("YOLOv8 object detection + ResNet50 features + LSTM beam-search captioning")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Loading models and generating caption..."):
            try:
                yolo_model, resnet_model, tokenizer, lstm_model = load_models()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    img_feature, yolo_results, detected_classes = extract_image_features(
                        tmp_path, yolo_model, resnet_model
                    )
                    caption = generate_caption_beam_search(
                        lstm_model, tokenizer, img_feature, MAX_LENGTH,
                        detected_classes, beam_width=5, alpha=0.8
                    )
                    st.success(caption.capitalize() + ".")
                    if detected_classes:
                        st.write("**Detected objects:**", ", ".join(sorted(set(detected_classes))))
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
else:
    st.info("Upload an image to generate a caption.")
