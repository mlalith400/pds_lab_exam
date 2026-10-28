# ================================================================
# APPLE PLANT DISEASE CLASSIFICATION
# Models: InceptionV3 and Xception
# Complete single-cell Kaggle implementation
# ================================================================

import os
import gc
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize

from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger
)

warnings.filterwarnings("ignore")

# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

DATASET_DIR = "/kaggle/input/competitions/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train_images")

OUTPUT_DIR = "/kaggle/working/apple_disease_inception_xception"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("APPLE PLANT DISEASE CLASSIFICATION")
print("InceptionV3 and Xception")
print("=" * 70)
print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("Available GPUs:", len(gpus))

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

if not os.path.isfile(TRAIN_CSV):
    raise FileNotFoundError(f"CSV not found: {TRAIN_CSV}")

if not os.path.isdir(TRAIN_IMAGES_DIR):
    raise FileNotFoundError(f"Image directory not found: {TRAIN_IMAGES_DIR}")

# ================================================================
# 2. LOAD DATASET
# ================================================================

df = pd.read_csv(TRAIN_CSV)

if not {"image", "labels"}.issubset(df.columns):
    raise ValueError(
        f"Expected image and labels columns. Found: {list(df.columns)}"
    )

df["image_path"] = df["image"].apply(
    lambda name: os.path.join(TRAIN_IMAGES_DIR, name)
)

missing_mask = ~df["image_path"].apply(os.path.isfile)

if missing_mask.any():
    print("Missing images:", int(missing_mask.sum()))
    df = df.loc[~missing_mask].copy()

df["labels"] = df["labels"].astype(str).str.strip()
df = df[df["labels"] != ""].reset_index(drop=True)

if df.empty:
    raise ValueError("No valid images found.")

# Treat each complete label string as one class.
df["class_name"] = df["labels"]

class_names = sorted(df["class_name"].unique())
NUM_CLASSES = len(class_names)

class_to_index = {
    name: idx for idx, name in enumerate(class_names)
}

index_to_class = {
    idx: name for name, idx in class_to_index.items()
}

df["label_id"] = df["class_name"].map(class_to_index).astype(int)

print("\nImages:", len(df))
print("Classes:", NUM_CLASSES)
print("\nClass distribution:")
print(df["class_name"].value_counts())

with open(
    os.path.join(OUTPUT_DIR, "class_mapping.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(class_to_index, f, indent=4, ensure_ascii=False)

# ================================================================
# 3. STRATIFIED DATA SPLIT
# ================================================================

class_counts = df["label_id"].value_counts()

if (class_counts < 3).any():
    raise ValueError(
        "At least one class has fewer than 3 images. "
        "A stratified train/validation/test split cannot represent "
        "every class safely. Review the rare classes."
    )

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["label_id"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))

# ================================================================
# 4. CLASS WEIGHTS
# ================================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label_id"].to_numpy()
)

class_weights = {
    int(i): float(w) for i, w in enumerate(weights)
}

# ================================================================
# 5. DATA PIPELINE
# ================================================================

AUTOTUNE = tf.data.AUTOTUNE

def decode_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE,
        antialias=True
    )

    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.int32)


def make_dataset(dataframe, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dataframe["image_path"].astype(str).to_numpy(),
            dataframe["label_id"].to_numpy(dtype=np.int32)
        )
    )

    if training:
        dataset = dataset.shuffle(
            min(len(dataframe), 5000),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# ================================================================
# 6. DATA AUGMENTATION
# ================================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10)
    ],
    name="data_augmentation"
)

# ================================================================
# 7. INCEPTIONV3
# ================================================================

def build_inceptionv3():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    x = augmentation(inputs)

    # InceptionV3 preprocessing maps pixels to approximately [-1, 1].
    x = tf.keras.applications.inception_v3.preprocess_input(x)

    backbone = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )

    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="InceptionV3"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ================================================================
# 8. XCEPTION
# ================================================================

def build_xception():
    inputs = layers.Input(shape=(*IMG_SIZE, 3))

    x = augmentation(inputs)

    # Xception preprocessing maps pixels to approximately [-1, 1].
    x = tf.keras.applications.xception.preprocess_input(x)

    backbone = Xception(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )

    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="Xception"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ================================================================
# 9. TRAINING AND EVALUATION
# ================================================================

def train_and_evaluate(model_name, builder):
    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    model_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f"{model_name.lower()}_best.keras"
    )

    model = builder()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            os.path.join(model_dir, "training_history.csv")
        )
    ]

    start_time = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    training_seconds = time.time() - start_time

    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    test_loss, _ = model.evaluate(test_ds, verbose=1)

    inference_start = time.perf_counter()
    probabilities = model.predict(test_ds, verbose=1)
    inference_seconds = time.perf_counter() - inference_start

    y_true = test_df["label_id"].to_numpy(dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    try:
        if NUM_CLASSES == 2:
            auc = roc_auc_score(y_true, probabilities[:, 1])
        else:
            binary_labels = label_binarize(
                y_true,
                classes=np.arange(NUM_CLASSES)
            )

            auc = roc_auc_score(
                binary_labels,
                probabilities,
                average="macro",
                multi_class="ovr"
            )
    except Exception as error:
        print("ROC-AUC calculation failed:", error)
        auc = np.nan

    parameters = model.count_params()

    inference_ms = (
        inference_seconds / max(len(test_df), 1)
    ) * 1000

    throughput = (
        len(test_df) / inference_seconds
        if inference_seconds > 0 else 0
    )

    print("\nTEST METRICS")
    print("-" * 55)
    print(f"Test loss:               {test_loss:.4f}")
    print(f"Accuracy:                {accuracy:.4f}")
    print(f"Macro precision:         {precision:.4f}")
    print(f"Macro recall:            {recall:.4f}")
    print(f"Macro F1-score:          {f1:.4f}")
    print(f"Macro ROC-AUC:           {auc:.4f}")
    print(f"Parameters:              {parameters:,}")
    print(f"Training time:           {training_seconds:.2f} seconds")
    print(f"Inference ms/image:      {inference_ms:.3f}")
    print(f"Inference images/second: {throughput:.3f}")

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=[
            index_to_class[i] for i in range(NUM_CLASSES)
        ],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(model_dir, "classification_report.csv")
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            target_names=[
                index_to_class[i] for i in range(NUM_CLASSES)
            ],
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    plt.figure(
        figsize=(
            max(10, NUM_CLASSES * 0.7),
            max(8, NUM_CLASSES * 0.55)
        )
    )

    sns.heatmap(
        cm,
        annot=NUM_CLASSES <= 15,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            index_to_class[i] for i in range(NUM_CLASSES)
        ],
        yticklabels=[
            index_to_class[i] for i in range(NUM_CLASSES)
        ]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        os.path.join(model_dir, "confusion_matrix.png"),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # Training curves
    pd.DataFrame(history.history).to_csv(
        os.path.join(model_dir, "training_curves.csv"),
        index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title(f"{model_name} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title(f"{model_name} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(model_dir, "training_curves.png"),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # Save predictions
    predictions_df = test_df[
        ["image", "labels", "label_id"]
    ].copy()

    predictions_df["predicted_label_id"] = y_pred

    predictions_df["predicted_class"] = [
        index_to_class[int(i)] for i in y_pred
    ]

    predictions_df["correct"] = (
        predictions_df["label_id"]
        == predictions_df["predicted_label_id"]
    )

    predictions_df.to_csv(
        os.path.join(model_dir, "test_predictions.csv"),
        index=False
    )

    results = {
        "Model": model_name,
        "Test_Loss": float(test_loss),
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "ROC_AUC": float(auc),
        "Parameters": int(parameters),
        "Parameters_M": round(parameters / 1_000_000, 3),
        "Training_Time_sec": round(training_seconds, 2),
        "Inference_ms_per_image": round(inference_ms, 4),
        "Inference_images_sec": round(throughput, 4)
    }

    del model, probabilities, history
    gc.collect()
    tf.keras.backend.clear_session()

    return results

# ================================================================
# 10. TRAIN BOTH MODELS
# ================================================================

inception_results = train_and_evaluate(
    "InceptionV3",
    build_inceptionv3
)

xception_results = train_and_evaluate(
    "Xception",
    build_xception
)

# ================================================================
# 11. FINAL COMPARISON
# ================================================================

comparison_df = pd.DataFrame(
    [inception_results, xception_results]
).sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\n" + "=" * 95)
print("FINAL COMPARISON: INCEPTIONV3 VS XCEPTION")
print("=" * 95)

print(
    comparison_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
            "Parameters_M",
            "Training_Time_sec",
            "Inference_ms_per_image",
            "Inference_images_sec"
        ]
    ].to_string(index=False)
)

# Metrics comparison chart
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

comparison_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    rot=0
)

plt.title("InceptionV3 vs Xception - Test Metrics")
plt.ylabel("Score")
plt.xlabel("Model")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "metrics_comparison.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Parameter comparison
plt.figure(figsize=(8, 5))

plt.bar(
    comparison_df["Model"],
    comparison_df["Parameters_M"]
)

plt.title("Model Parameter Comparison")
plt.ylabel("Parameters (millions)")
plt.xlabel("Model")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "parameter_comparison.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Best model summary
best = comparison_df.iloc[0]

print("\nBEST MODEL BY TEST ACCURACY")
print("-" * 45)
print("Model:", best["Model"])
print(f"Accuracy:  {best['Accuracy'] * 100:.2f}%")
print(f"Precision: {best['Precision'] * 100:.2f}%")
print(f"Recall:    {best['Recall'] * 100:.2f}%")
print(f"F1-score:  {best['F1_Score'] * 100:.2f}%")
print(f"ROC-AUC:   {best['ROC_AUC']:.4f}")

print("\nResults saved to:", OUTPUT_DIR)
print("Training and evaluation completed.")