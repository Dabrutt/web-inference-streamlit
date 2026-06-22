from __future__ import annotations

import base64
import html
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "models" / "best.pt"
HERO_IMAGE_PATH = APP_DIR / "assets" / "hero.jpg"

st.set_page_config(
    page_title="Trashification - Deteksi Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INFERENCE_LOCK = threading.Lock()

# IoU tetap digunakan secara internal untuk mengurangi bounding box duplikat.
# Pengguna umum tidak perlu mengaturnya dari antarmuka.
FIXED_IOU_THRESHOLD = 0.50


# =========================================================
# 2. DAFTAR CLASS DAN KELOMPOK SAMPAH
# =========================================================
CLASS_NAMES = [
    "battery",
    "can",
    "cardboard_bowl",
    "cardboard_box",
    "chemical_plastic_bottle",
    "chemical_plastic_gallon",
    "chemical_spray_can",
    "light_bulb",
    "paint_bucket",
    "plastic_bag",
    "plastic_bottle",
    "plastic_bottle_cap",
    "plastic_box",
    "plastic_cultery",
    "plastic_cup",
    "plastic_cup_lid",
    "reuseable_paper",
    "scrap_paper",
    "scrap_plastic",
    "snack_bag",
    "stick",
    "straw",
]

CATEGORY_CLASSES = {
    "Recyclable": {
        "can",
        "cardboard_bowl",
        "cardboard_box",
        "plastic_bottle",
        "plastic_bottle_cap",
        "plastic_box",
        "plastic_cup",
        "plastic_cup_lid",
        "reuseable_paper",
        "scrap_paper",
        "scrap_plastic",
    },
    "Biodegradable": {
        "stick",
    },
    "Hazardous": {
        "battery",
        "chemical_plastic_bottle",
        "chemical_plastic_gallon",
        "chemical_spray_can",
        "light_bulb",
        "paint_bucket",
    },
    "Residual": {
        "plastic_bag",
        "plastic_cultery",
        "snack_bag",
        "straw",
    },
}

CLASS_TO_CATEGORY = {
    class_name: category
    for category, class_names in CATEGORY_CLASSES.items()
    for class_name in class_names
}

CATEGORY_META = {
    "Recyclable": {
        "icon": "♻️",
        "description": "Sampah yang berpotensi diproses dan digunakan kembali.",
    },
    "Biodegradable": {
        "icon": "🌿",
        "description": "Sampah yang relatif dapat terurai secara alami.",
    },
    "Hazardous": {
        "icon": "⚠️",
        "description": "Sampah yang memerlukan penanganan khusus.",
    },
    "Residual": {
        "icon": "🗑️",
        "description": "Sampah yang sulit atau tidak umum didaur ulang.",
    },
}


# Pembagian sampah berdasarkan sifat asalnya.
# Pada 22 class dataset ini, "stick" menjadi satu-satunya class organik.
ORGANIC_CLASSES = {
    "stick",
}

CLASS_TO_WASTE_TYPE = {
    class_name: (
        "Organik"
        if class_name in ORGANIC_CLASSES
        else "Anorganik"
    )
    for class_name in CLASS_NAMES
}

WASTE_TYPE_META = {
    "Organik": {
        "icon": "🍃",
        "description": (
            "Sampah yang berasal dari makhluk hidup atau bahan alami "
            "dan umumnya lebih mudah terurai."
        ),
    },
    "Anorganik": {
        "icon": "🧱",
        "description": (
            "Sampah dari bahan nonhayati atau hasil proses industri yang "
            "umumnya membutuhkan waktu lebih lama untuk terurai."
        ),
    },
}

# Pengelompokan berdasarkan bahan dasar dominan.
# Mapping dapat disesuaikan lagi apabila karakter fisik dataset berbeda.
MATERIAL_CLASSES = {
    "Plastik": {
        "chemical_plastic_bottle",
        "chemical_plastic_gallon",
        "paint_bucket",
        "plastic_bag",
        "plastic_bottle",
        "plastic_bottle_cap",
        "plastic_box",
        "plastic_cultery",
        "plastic_cup",
        "plastic_cup_lid",
        "scrap_plastic",
        "snack_bag",
        "straw",
    },
    "Logam": {
        "can",
        "chemical_spray_can",
    },
    "Kertas/Karton": {
        "cardboard_bowl",
        "cardboard_box",
        "reuseable_paper",
        "scrap_paper",
    },
    "Kaca": {
        "light_bulb",
    },
    "Kayu": {
        "stick",
    },
    "Baterai": {
        "battery",
    },
}

CLASS_TO_MATERIAL = {
    class_name: material
    for material, class_names in MATERIAL_CLASSES.items()
    for class_name in class_names
}

MATERIAL_META = {
    "Plastik": {"icon": "🧴"},
    "Logam": {"icon": "🥫"},
    "Kertas/Karton": {"icon": "📦"},
    "Kaca": {"icon": "💡"},
    "Kayu": {"icon": "🪵"},
    "Baterai": {"icon": "🔋"},
}

GROUP_EXPLANATION_META = {
    "Organik": WASTE_TYPE_META["Organik"],
    "Anorganik": WASTE_TYPE_META["Anorganik"],
    "Recyclable": {
        "icon": CATEGORY_META["Recyclable"]["icon"],
        "description": (
            "Sampah yang masih dapat dipilah, diolah, dan dimanfaatkan "
            "kembali menjadi bahan atau produk baru."
        ),
    },
    "Biodegradable": {
        "icon": CATEGORY_META["Biodegradable"]["icon"],
        "description": (
            "Sampah yang dapat diuraikan oleh mikroorganisme secara alami "
            "dalam kondisi lingkungan yang sesuai."
        ),
    },
    "Hazardous": {
        "icon": CATEGORY_META["Hazardous"]["icon"],
        "description": (
            "Sampah yang mengandung bahan berbahaya dan memerlukan "
            "penyimpanan serta penanganan khusus."
        ),
    },
    "Residual": {
        "icon": CATEGORY_META["Residual"]["icon"],
        "description": (
            "Sampah yang sulit didaur ulang atau dikomposkan sehingga "
            "biasanya menuju pengolahan atau pembuangan akhir."
        ),
    },
}


# =========================================================
# 3. STYLE ANTARMUKA
# =========================================================
def get_hero_background() -> str:
    """Gunakan assets/hero.jpg bila tersedia, selain itu gunakan gradient."""
    if HERO_IMAGE_PATH.exists():
        encoded = base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("utf-8")
        return (
            "linear-gradient(90deg, rgba(7, 34, 24, .88), "
            "rgba(7, 34, 24, .30)), "
            f"url('data:image/jpeg;base64,{encoded}')"
        )

    return (
        "radial-gradient(circle at 78% 34%, rgba(71, 180, 137, .45), "
        "transparent 27%), "
        "linear-gradient(125deg, #062d22 0%, #0b4c3b 48%, #148a6e 100%)"
    )


st.markdown(
    f"""
    <style>
        :root {{
            --green: #138a6f;
            --green-dark: #075a47;
            --green-soft: #e8f4f0;
            --ink: #14211d;
            --muted: #62706a;
            --panel: #ffffff;
            --page: #f3f5f4;
            --line: #dfe7e3;
        }}

        .stApp {{
            background: var(--page);
        }}

        .block-container {{
            max-width: 1320px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }}

        .brand-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0 0 1rem 0;
        }}

        .brand {{
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            color: var(--green);
        }}

        .brand-nav {{
            color: var(--green-dark);
            font-weight: 750;
            word-spacing: 2rem;
        }}

        .hero {{
            min-height: 390px;
            border-radius: 30px;
            padding: 3.2rem;
            display: flex;
            align-items: center;
            overflow: hidden;
            background-image: {get_hero_background()};
            background-size: cover;
            background-position: center;
            box-shadow: 0 16px 45px rgba(7, 52, 38, .18);
            margin-bottom: 3rem;
        }}

        .hero h1 {{
            color: white;
            font-size: clamp(3.5rem, 8vw, 7.8rem);
            line-height: .82;
            letter-spacing: -0.07em;
            margin: 0;
            max-width: 760px;
            text-transform: uppercase;
            text-shadow: 0 4px 24px rgba(0, 0, 0, .24);
        }}

        .section-heading {{
            text-align: center;
            color: var(--green);
            font-weight: 900;
            font-size: 2.2rem;
            letter-spacing: -0.04em;
            margin: 0 0 1.6rem 0;
        }}

        .section-heading::after {{
            content: "";
            display: block;
            width: 235px;
            max-width: 70%;
            height: 4px;
            border-radius: 999px;
            background: var(--green);
            margin: .35rem auto 0;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--panel);
            border-color: var(--line);
            border-radius: 24px;
            box-shadow: 0 10px 28px rgba(20, 33, 29, .08);
        }}

        div.stButton > button {{
            border-radius: 12px;
            min-height: 44px;
            font-weight: 800;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid var(--line);
            padding: 1rem;
            border-radius: 16px;
            box-shadow: 0 6px 20px rgba(20, 33, 29, .06);
        }}

        .camera-placeholder {{
            height: 285px;
            border-radius: 18px;
            background:
                linear-gradient(135deg, rgba(19, 138, 111, .08), rgba(19, 138, 111, .02)),
                #eef1ef;
            border: 2px dashed #b7c6c0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #82908b;
            text-align: center;
            margin-bottom: .8rem;
        }}

        .camera-placeholder .icon {{
            font-size: 4rem;
            margin-bottom: .55rem;
            opacity: .65;
        }}

        .group-info-card {{
            background: white;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.15rem 1.2rem;
            min-height: 175px;
            box-shadow: 0 8px 24px rgba(20, 33, 29, .07);
            margin-bottom: 1rem;
        }}

        .group-info-title {{
            color: var(--green);
            font-weight: 900;
            font-size: 1.1rem;
            margin-bottom: .55rem;
        }}

        .group-info-description {{
            color: var(--muted);
            font-size: .94rem;
            line-height: 1.55;
        }}

        .category-card {{
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.05rem 1.15rem;
            min-height: 145px;
            box-shadow: 0 7px 20px rgba(20, 33, 29, .06);
            margin-bottom: .8rem;
        }}

        .compact-card {{
            min-height: 128px;
        }}

        .category-title {{
            color: var(--green);
            font-weight: 900;
            font-size: 1.05rem;
            margin-bottom: .35rem;
        }}

        .category-count {{
            font-size: 1.8rem;
            line-height: 1;
            color: var(--ink);
            font-weight: 900;
            margin-bottom: .55rem;
        }}

        .category-labels {{
            color: var(--muted);
            font-size: .89rem;
            line-height: 1.45;
        }}

        .status-ok {{
            background: #eaf8f2;
            color: #086347;
            padding: .7rem .8rem;
            border-radius: 12px;
            border: 1px solid #bee7d7;
            font-weight: 700;
        }}

        .status-error {{
            background: #fff1f0;
            color: #9d2c24;
            padding: .7rem .8rem;
            border-radius: 12px;
            border: 1px solid #f0c8c4;
            font-weight: 700;
        }}

        .small-note {{
            color: var(--muted);
            font-size: .88rem;
            line-height: 1.45;
        }}

        @media (max-width: 800px) {{
            .hero {{
                min-height: 290px;
                padding: 2rem;
            }}
            .hero h1 {{
                font-size: 4rem;
            }}
            .brand-nav {{
                display: none;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. FUNGSI MODEL DAN INFERENSI
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def resolve_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))

    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])

    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]

    return f"class_{class_id}"


def extract_detections(result: Any) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confidences = result.boxes.conf.detach().cpu().numpy().tolist()
    coordinates = result.boxes.xyxy.detach().cpu().numpy().tolist()

    for class_id, confidence, bbox in zip(
        class_ids,
        confidences,
        coordinates,
        strict=False,
    ):
        class_name = resolve_class_name(result.names, class_id).strip().lower()
        category = CLASS_TO_CATEGORY.get(class_name, "Residual")
        waste_type = CLASS_TO_WASTE_TYPE.get(class_name, "Anorganik")
        material = CLASS_TO_MATERIAL.get(class_name, "Lainnya")

        detections.append(
            {
                "class_id": class_id,
                "label": class_name,
                "confidence": float(confidence),
                "waste_type": waste_type,
                "category": category,
                "material": material,
                "bbox": [round(float(value), 2) for value in bbox],
            }
        )

    return detections


def run_image_inference(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    image_size: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    rgb_image = image.convert("RGB")

    with INFERENCE_LOCK:
        results = model.predict(
            source=rgb_image,
            conf=confidence,
            iou=FIXED_IOU_THRESHOLD,
            imgsz=image_size,
            verbose=False,
            max_det=300,
        )

    result = results[0]
    detections = extract_detections(result)

    annotated_bgr = result.plot(
        labels=True,
        conf=True,
        line_width=2,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(annotated_rgb), detections


def summarize_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    total_objects = len(detections)
    unique_labels = sorted({item["label"] for item in detections})

    waste_type_counts = Counter(item["waste_type"] for item in detections)
    waste_type_label_counts: dict[str, Counter] = {
        waste_type: Counter() for waste_type in WASTE_TYPE_META
    }

    category_counts = Counter(item["category"] for item in detections)
    category_label_counts: dict[str, Counter] = {
        category: Counter() for category in CATEGORY_META
    }

    material_counts = Counter(item["material"] for item in detections)
    material_label_counts: dict[str, Counter] = {
        material: Counter() for material in MATERIAL_META
    }

    for item in detections:
        waste_type_label_counts[item["waste_type"]][item["label"]] += 1
        category_label_counts[item["category"]][item["label"]] += 1

        material = item["material"]
        if material not in material_label_counts:
            material_label_counts[material] = Counter()
        material_label_counts[material][item["label"]] += 1

    average_confidence = (
        sum(item["confidence"] for item in detections) / total_objects
        if total_objects
        else 0.0
    )

    dominant_category = (
        category_counts.most_common(1)[0][0] if category_counts else "-"
    )

    return {
        "total_objects": total_objects,
        "unique_labels": unique_labels,
        "unique_count": len(unique_labels),
        "waste_type_counts": waste_type_counts,
        "waste_type_label_counts": waste_type_label_counts,
        "category_counts": category_counts,
        "category_label_counts": category_label_counts,
        "material_counts": material_counts,
        "material_label_counts": material_label_counts,
        "average_confidence": average_confidence,
        "dominant_category": dominant_category,
    }


def draw_realtime_overlay(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    summary = summarize_detections(detections)
    _, width = frame_bgr.shape[:2]

    panel_width = min(620, max(390, width - 20))
    panel_height = 150

    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (10, 35, 27),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0, frame_bgr)

    waste_type_counts = summary["waste_type_counts"]
    category_counts = summary["category_counts"]

    lines = [
        (
            f"Total: {summary['total_objects']} | "
            f"Label unik: {summary['unique_count']}"
        ),
        (
            f"Organik: {waste_type_counts.get('Organik', 0)} | "
            f"Anorganik: {waste_type_counts.get('Anorganik', 0)}"
        ),
        (
            f"Recyclable: {category_counts.get('Recyclable', 0)} | "
            f"Biodegradable: {category_counts.get('Biodegradable', 0)}"
        ),
        (
            f"Hazardous: {category_counts.get('Hazardous', 0)} | "
            f"Residual: {category_counts.get('Residual', 0)}"
        ),
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame_bgr,
            line,
            (24, 42 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame_bgr


class RealtimeFrameProcessor:
    """Callable untuk memproses frame kamera secara realtime."""

    def __init__(
        self,
        model: YOLO,
        confidence: float,
        image_size: int,
        frame_interval: int,
    ) -> None:
        self.model = model
        self.confidence = confidence
        self.image_size = image_size
        self.frame_interval = max(1, frame_interval)
        self.frame_number = 0
        self.last_annotated_frame: np.ndarray | None = None

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        frame_bgr = frame.to_ndarray(format="bgr24")
        self.frame_number += 1

        # Untuk CPU, interval > 1 dapat mengurangi beban inferensi.
        if (
            self.frame_interval > 1
            and self.frame_number % self.frame_interval != 0
            and self.last_annotated_frame is not None
        ):
            return av.VideoFrame.from_ndarray(
                self.last_annotated_frame,
                format="bgr24",
            )

        with INFERENCE_LOCK:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.confidence,
                iou=FIXED_IOU_THRESHOLD,
                imgsz=self.image_size,
                verbose=False,
                max_det=300,
            )

        result = results[0]
        detections = extract_detections(result)
        annotated_bgr = result.plot(
            labels=True,
            conf=True,
            line_width=2,
        )
        annotated_bgr = draw_realtime_overlay(
            annotated_bgr,
            detections,
        )

        self.last_annotated_frame = annotated_bgr
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


# =========================================================
# 5. KOMPONEN TAMPILAN HASIL
# =========================================================
def render_group_info_card(group_name: str) -> None:
    meta = GROUP_EXPLANATION_META[group_name]

    st.markdown(
        f"""
        <div class="group-info-card">
            <div class="group-info-title">
                {meta["icon"]} {html.escape(group_name.upper())}
            </div>
            <div class="group-info-description">
                {html.escape(meta["description"])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_group_card(
    group_name: str,
    count: int,
    label_counts: Counter,
    meta: dict[str, Any],
    compact: bool = False,
) -> None:
    if label_counts:
        label_text = "<br>".join(
            f"{html.escape(label)} <strong>×{amount}</strong>"
            for label, amount in label_counts.most_common()
        )
    else:
        label_text = "Tidak ada sampah dari kelompok ini."

    card_class = "category-card compact-card" if compact else "category-card"

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="category-title">
                {meta.get("icon", "•")} {html.escape(group_name.upper())}
            </div>
            <div class="category-count">{count}</div>
            <div class="category-labels">{label_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detection_results(
    annotated_image: Image.Image,
    detections: list[dict[str, Any]],
) -> None:
    summary = summarize_detections(detections)

    st.image(
        annotated_image,
        caption="Hasil deteksi YOLO",
        use_container_width=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Jumlah sampah",
        summary["total_objects"],
    )
    metric_columns[1].metric(
        "Label unik",
        summary["unique_count"],
    )
    metric_columns[2].metric(
        "Rata-rata confidence",
        f"{summary['average_confidence'] * 100:.1f}%",
    )
    metric_columns[3].metric(
        "Kategori dominan",
        summary["dominant_category"],
    )

    st.markdown("#### Berdasarkan Sifat Sampah")
    waste_type_columns = st.columns(2)

    for column, waste_type in zip(
        waste_type_columns,
        WASTE_TYPE_META,
        strict=False,
    ):
        with column:
            render_result_group_card(
                group_name=waste_type,
                count=summary["waste_type_counts"].get(waste_type, 0),
                label_counts=summary["waste_type_label_counts"][waste_type],
                meta=WASTE_TYPE_META[waste_type],
            )

    st.markdown("#### Berdasarkan Kategori Penanganan")

    category_row_one = st.columns(2)
    category_row_two = st.columns(2)

    category_positions = [
        (category_row_one[0], "Recyclable"),
        (category_row_one[1], "Biodegradable"),
        (category_row_two[0], "Hazardous"),
        (category_row_two[1], "Residual"),
    ]

    for column, category in category_positions:
        with column:
            render_result_group_card(
                group_name=category,
                count=summary["category_counts"].get(category, 0),
                label_counts=summary["category_label_counts"][category],
                meta=CATEGORY_META[category],
            )

    st.markdown("#### Berdasarkan Bahan Dasar")

    material_names = list(MATERIAL_META)
    material_rows = [
        st.columns(3),
        st.columns(3),
    ]

    for index, material in enumerate(material_names):
        row_index = index // 3
        column_index = index % 3

        with material_rows[row_index][column_index]:
            render_result_group_card(
                group_name=material,
                count=summary["material_counts"].get(material, 0),
                label_counts=summary["material_label_counts"][material],
                meta=MATERIAL_META[material],
                compact=True,
            )

    if detections:
        detail_rows = [
            {
                "No.": index,
                "Label": item["label"],
                "Sifat": item["waste_type"],
                "Bahan Dasar": item["material"],
                "Kategori Penanganan": item["category"],
                "Confidence": f"{item['confidence'] * 100:.2f}%",
            }
            for index, item in enumerate(detections, start=1)
        ]

        with st.expander("Lihat detail seluruh objek yang terdeteksi"):
            st.dataframe(
                pd.DataFrame(detail_rows),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info(
            "Tidak ada objek yang melewati nilai confidence yang dipilih. "
            "Coba turunkan threshold atau gunakan gambar yang lebih jelas."
        )


# =========================================================
# 6. SESSION STATE
# =========================================================
DEFAULT_SESSION_VALUES = {
    "input_mode": "Impor Foto",
    "realtime_active": False,
    "annotated_image": None,
    "detections": None,
    "realtime_processor": None,
    "realtime_processor_key": None,
}

for key, value in DEFAULT_SESSION_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_static_result() -> None:
    st.session_state.annotated_image = None
    st.session_state.detections = None


# =========================================================
# 7. SIDEBAR DAN PEMUATAN MODEL
# =========================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")

    model_path_input = st.text_input(
        "Lokasi model YOLO",
        value=os.getenv("YOLO_MODEL_PATH", str(DEFAULT_MODEL_PATH)),
        help="Contoh: models/best.pt",
    )

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )


    image_size = st.select_slider(
        "Ukuran inferensi",
        options=[320, 416, 512, 640, 768],
        value=640,
        help="Ukuran lebih kecil biasanya lebih cepat, tetapi dapat mengurangi ketelitian.",
    )

    frame_interval = st.slider(
        "Proses setiap N frame realtime",
        min_value=1,
        max_value=5,
        value=2,
        help="Gunakan 1 untuk hasil paling halus atau 2–5 untuk meringankan CPU.",
    )

    st.divider()
    st.caption(
        "Mapping sifat, kategori penanganan, dan bahan dasar dapat diubah di app.py."
    )

model_path = Path(model_path_input).expanduser()

model: YOLO | None = None
model_error: str | None = None

if not model_path.exists():
    model_error = (
        f"Model tidak ditemukan di: {model_path}. "
        "Letakkan file best.pt pada folder models."
    )
else:
    try:
        with st.spinner("Memuat model YOLO..."):
            model = load_model(str(model_path))
    except Exception as exc:
        model_error = f"Model gagal dimuat: {exc}"

with st.sidebar:
    if model is not None:
        st.markdown(
            '<div class="status-ok">✅ Model berhasil dimuat</div>',
            unsafe_allow_html=True,
        )

        model_names = getattr(model, "names", {})
        if isinstance(model_names, dict):
            loaded_names = [str(model_names[key]) for key in sorted(model_names)]
        else:
            loaded_names = [str(name) for name in model_names]

        if loaded_names and loaded_names != CLASS_NAMES:
            st.warning(
                "Nama atau urutan class pada model berbeda dari daftar class "
                "di aplikasi. Pastikan model memakai 22 class yang sama."
            )
    else:
        st.markdown(
            f'<div class="status-error">❌ {html.escape(model_error or "Model belum tersedia")}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Daftar 22 class"):
        for class_id, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_id}. `{class_name}`")


# =========================================================
# 8. HEADER, HERO, DAN PENJELASAN KELOMPOK
# =========================================================
st.markdown(
    """
    <div class="brand-row">
        <div class="brand">TRASHIFICATION</div>
        <div class="brand-nav">HOME&nbsp;&nbsp;&nbsp;&nbsp;DETECT</div>
    </div>
    <div class="hero">
        <h1>Trash<br>Detection</h1>
    </div>
    <div class="section-heading">KENALI KELOMPOK SAMPAH</div>
    """,
    unsafe_allow_html=True,
)

explanation_names = list(GROUP_EXPLANATION_META)
explanation_rows = [
    st.columns(3),
    st.columns(3),
]

for index, group_name in enumerate(explanation_names):
    row_index = index // 3
    column_index = index % 3

    with explanation_rows[row_index][column_index]:
        render_group_info_card(group_name)

st.markdown(
    """
    <div style="height: 1.4rem;"></div>
    <div class="section-heading">DETECT YOUR TRASH</div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 9. AREA UTAMA: INPUT KIRI DAN HASIL KANAN
# =========================================================
left_column, right_column = st.columns([0.38, 0.62], gap="large")

with left_column:
    with st.container(border=True):
        st.subheader("📷 Sumber Gambar")

        mode_columns = st.columns(2)

        if mode_columns[0].button(
            "Ambil Foto",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.input_mode == "Ambil Foto"
                else "secondary"
            ),
        ):
            st.session_state.input_mode = "Ambil Foto"
            clear_static_result()
            st.rerun()

        if mode_columns[1].button(
            "Impor Foto",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.input_mode == "Impor Foto"
                else "secondary"
            ),
        ):
            st.session_state.input_mode = "Impor Foto"
            clear_static_result()
            st.rerun()

        selected_file = None

        if st.session_state.input_mode == "Ambil Foto":
            selected_file = st.camera_input(
                "Ambil gambar menggunakan kamera",
                key="camera_input",
            )
        else:
            selected_file = st.file_uploader(
                "Pilih foto sampah",
                type=["jpg", "jpeg", "png", "webp"],
                key="photo_uploader",
            )

        selected_image: Image.Image | None = None

        if selected_file is not None:
            try:
                selected_file.seek(0)
                selected_image = Image.open(selected_file).convert("RGB")
                st.image(
                    selected_image,
                    caption="Gambar yang dipilih",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Gambar tidak dapat dibaca: {exc}")
        else:
            st.markdown(
                """
                <div class="camera-placeholder">
                    <div class="icon">📸</div>
                    <strong>Belum ada gambar</strong>
                    <span>Ambil foto atau impor gambar dari perangkat.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        detect_clicked = st.button(
            "▶️ Deteksi Langsung",
            use_container_width=True,
            type="primary",
            disabled=(model is None or selected_image is None),
        )

        realtime_button_label = (
            "⏹️ Matikan Kamera Realtime"
            if st.session_state.realtime_active
            else "🎥 Aktifkan Kamera Realtime"
        )

        if st.button(
            realtime_button_label,
            use_container_width=True,
            disabled=(model is None),
        ):
            st.session_state.realtime_active = (
                not st.session_state.realtime_active
            )
            st.session_state.realtime_processor = None
            st.session_state.realtime_processor_key = None
            st.rerun()

        if model_error:
            st.caption(model_error)

        if detect_clicked and model is not None and selected_image is not None:
            try:
                with st.spinner("YOLO sedang mendeteksi sampah..."):
                    annotated_image, detections = run_image_inference(
                        model=model,
                        image=selected_image,
                        confidence=confidence_threshold,
                        image_size=image_size,
                    )

                st.session_state.annotated_image = annotated_image
                st.session_state.detections = detections
                st.success("Deteksi selesai.")
            except Exception as exc:
                st.exception(exc)

with right_column:
    with st.container(border=True):
        st.subheader("📊 Hasil Deteksi")

        if (
            st.session_state.annotated_image is not None
            and st.session_state.detections is not None
        ):
            render_detection_results(
                annotated_image=st.session_state.annotated_image,
                detections=st.session_state.detections,
            )
        else:
            st.markdown(
                """
                <div class="camera-placeholder">
                    <div class="icon">♻️</div>
                    <strong>Hasil deteksi akan tampil di sini</strong>
                    <span>
                        Sistem akan menampilkan bounding box, jumlah objek,
                        label unik, confidence, dan kelompok sampah.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "Setelah deteksi, hasil akan dikelompokkan berdasarkan sifat "
                "organik/anorganik, kategori penanganan, dan bahan dasarnya."
            )


# =========================================================
# 10. KAMERA REALTIME
# =========================================================
if st.session_state.realtime_active and model is not None:
    st.markdown("---")
    st.markdown(
        '<div class="section-heading">REALTIME DETECTION</div>',
        unsafe_allow_html=True,
    )

    processor_key = (
        str(model_path.resolve()),
        confidence_threshold,
        image_size,
        frame_interval,
    )

    if st.session_state.realtime_processor_key != processor_key:
        st.session_state.realtime_processor = RealtimeFrameProcessor(
            model=model,
            confidence=confidence_threshold,
            image_size=image_size,
            frame_interval=frame_interval,
        )
        st.session_state.realtime_processor_key = processor_key

    realtime_left, realtime_right = st.columns([0.72, 0.28], gap="large")

    with realtime_left:
        with st.container(border=True):
            webrtc_streamer(
                key="trash-realtime-camera",
                video_frame_callback=st.session_state.realtime_processor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 960},
                        "height": {"ideal": 540},
                    },
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                },
                desired_playing_state=True,
                async_processing=True,
            )

    with realtime_right:
        with st.container(border=True):
            st.subheader("Petunjuk Realtime")
            st.markdown(
                """
                1. Izinkan browser mengakses kamera.
                2. Arahkan kamera ke sampah.
                3. Bounding box dan statistik akan ditampilkan di video.
                4. Naikkan interval frame apabila deteksi terasa berat.
                """
            )
            st.info(
                "Kamera pada server online memerlukan HTTPS. "
                "Pada beberapa jaringan, koneksi WebRTC juga memerlukan TURN server."
            )


st.markdown("---")
st.caption(
    "Trashification • YOLO waste detection • "
    "Kategori pemilahan dapat disesuaikan dengan kebijakan setempat."
)
