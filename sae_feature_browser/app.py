from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from config import DATASET_LAYER_DIRS, TOP_K_FOR_PURITY


# ============================================================
# Session-state helpers
# ============================================================

def init_state(key, default_value):
    """Initialize a Streamlit session-state key only once."""
    if key not in st.session_state:
        st.session_state[key] = default_value


def keep_valid_selection(key, options, fallback=None):
    """
    Preserve the current value when it is still valid.

    If the current value is no longer present, use the fallback
    or the first available option.
    """
    options = list(options)

    if not options:
        return None

    if key not in st.session_state:
        if fallback is not None and fallback in options:
            st.session_state[key] = fallback
        else:
            st.session_state[key] = options[0]

    if st.session_state[key] not in options:
        if fallback is not None and fallback in options:
            st.session_state[key] = fallback
        else:
            st.session_state[key] = options[0]

    return st.session_state[key]


def feature_state_key(dataset_name: str, layer_name: str) -> str:
    """
    Generate a session-state key unique to one dataset/layer pair.

    This lets the app remember a different selected feature for:
    Caltech101 + mid_block,
    Caltech101 + up_ft1,
    Oxford Pets + mid_block,
    etc.
    """
    clean_dataset = dataset_name.replace(" ", "_").replace("-", "_")
    clean_layer = layer_name.replace(" ", "_").replace("-", "_")

    return f"selected_feature_{clean_dataset}_{clean_layer}"


def manual_feature_state_key(dataset_name: str, layer_name: str) -> str:
    clean_dataset = dataset_name.replace(" ", "_").replace("-", "_")
    clean_layer = layer_name.replace(" ", "_").replace("-", "_")

    return f"manual_feature_{clean_dataset}_{clean_layer}"


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_layer_data(
    dataset_name: str,
    layer_name: str,
    feature_dir_str: str,
    top_k: int,
):
    feature_dir = Path(feature_dir_str)

    required_files = [
        "max_activating_image_indices.pt",
        "max_activating_image_label_indices.pt",
        "max_activating_image_values.pt",
        "sae_mean_acts.pt",
        "sae_sparsity.pt",
        f"label_purity_top{top_k}.pt",
        f"majority_label_top{top_k}.pt",
        f"unique_label_count_top{top_k}.pt",
        f"feature_summary_top{top_k}.csv",
    ]

    missing = [
        filename
        for filename in required_files
        if not (feature_dir / filename).exists()
    ]

    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            f"Missing files in:\n{feature_dir}\n\n{missing_text}"
        )

    return {
        "dataset_name": dataset_name,
        "layer_name": layer_name,
        "feature_dir": feature_dir,
        "indices": torch.load(
            feature_dir / "max_activating_image_indices.pt",
            map_location="cpu",
        ),
        "labels": torch.load(
            feature_dir / "max_activating_image_label_indices.pt",
            map_location="cpu",
        ),
        "values": torch.load(
            feature_dir / "max_activating_image_values.pt",
            map_location="cpu",
        ),
        "mean_acts": torch.load(
            feature_dir / "sae_mean_acts.pt",
            map_location="cpu",
        ),
        "sparsity": torch.load(
            feature_dir / "sae_sparsity.pt",
            map_location="cpu",
        ),
        "purity": torch.load(
            feature_dir / f"label_purity_top{top_k}.pt",
            map_location="cpu",
        ),
        "majority_label": torch.load(
            feature_dir / f"majority_label_top{top_k}.pt",
            map_location="cpu",
        ),
        "unique_label_count": torch.load(
            feature_dir / f"unique_label_count_top{top_k}.pt",
            map_location="cpu",
        ),
        "summary": pd.read_csv(
            feature_dir / f"feature_summary_top{top_k}.csv"
        ),
    }


# ============================================================
# Image helpers
# ============================================================

def image_rank(path: Path):
    """
    Expected filename:

    rank_imageIndex_activationValue_label.png

    Example:

    0_2011_22.14_58.png
    """
    try:
        return int(path.stem.split("_")[0])
    except (ValueError, IndexError):
        return 999999


def get_feature_image_paths(feature_dir: Path, feature_id: int):
    feature_folder = feature_dir / str(feature_id)

    if not feature_folder.exists():
        return []

    return sorted(
        feature_folder.glob("*.png"),
        key=image_rank,
    )


def parse_image_filename(path: Path):
    """
    Parse:

    rank_imageIndex_activationValue_label.png
    """
    parts = path.stem.split("_")

    if len(parts) < 4:
        return None

    try:
        return {
            "rank": int(parts[0]),
            "image_index": int(parts[1]),
            "activation": float(parts[2]),
            "label": int(parts[3]),
        }
    except (ValueError, IndexError):
        return None


def show_image(image, caption: str):
    """
    Support both older and newer Streamlit versions.
    """
    try:
        st.image(
            image,
            caption=caption,
            use_container_width=True,
        )
    except TypeError:
        st.image(
            image,
            caption=caption,
            use_column_width=True,
        )


# ============================================================
# Table and ranking helpers
# ============================================================

def get_top_activation_table(
    data,
    feature_id: int,
    top_k_display: int,
):
    indices = data["indices"]
    labels = data["labels"]
    values = data["values"]

    rows = []
    max_k = min(top_k_display, values.shape[1])

    for rank in range(max_k):
        activation_value = float(
            values[feature_id, rank].item()
        )

        rows.append(
            {
                "rank": rank,
                "image_index": int(
                    indices[feature_id, rank].item()
                ),
                "activation": activation_value,
                "label": int(
                    labels[feature_id, rank].item()
                ),
                "positive": activation_value > 0,
            }
        )

    return pd.DataFrame(rows)


def select_feature_by_ranking(
    summary: pd.DataFrame,
    ranking_method: str,
    top_n: int,
):
    if ranking_method == "Mean activation":
        sorted_df = summary.sort_values(
            "mean_activation",
            ascending=False,
        )

    elif ranking_method == "Sparsity":
        sorted_df = summary.sort_values(
            "sparsity",
            ascending=False,
        )

    elif ranking_method == "Label purity":
        sorted_df = summary.sort_values(
            "label_purity",
            ascending=False,
        )

    elif ranking_method == "Class-specific score":
        sorted_df = summary.sort_values(
            "class_specific_score",
            ascending=False,
        )

    elif ranking_method == "Sparse class-specific score":
        sorted_df = summary.sort_values(
            "sparse_class_specific_score",
            ascending=False,
        )

    elif ranking_method == "Saved image folders only":
        sorted_df = summary[
            summary["has_saved_folder"] == True
        ].sort_values(
            "mean_activation",
            ascending=False,
        )

    else:
        sorted_df = summary.sort_values(
            "mean_activation",
            ascending=False,
        )

    return sorted_df.head(top_n)


# ============================================================
# Streamlit application
# ============================================================

st.set_page_config(
    page_title="k-SAE Feature Browser",
    layout="wide",
)

st.title("k-SAE Feature Browser")
st.caption(
    "Browse top activating images and statistics for "
    "Stable Diffusion k-SAE features."
)


# ============================================================
# Discover valid datasets and layer directories
# ============================================================

available_datasets = {}

for dataset_name, layer_mapping in DATASET_LAYER_DIRS.items():
    existing_layers = {
        layer_name: Path(layer_path)
        for layer_name, layer_path in layer_mapping.items()
        if Path(layer_path).exists()
    }

    if existing_layers:
        available_datasets[dataset_name] = existing_layers


if not available_datasets:
    st.error(
        "No valid dataset/layer directories were found. "
        "Please check DATASET_LAYER_DIRS in config.py."
    )
    st.stop()


dataset_options = list(available_datasets.keys())


# ============================================================
# Initialize shared session state
# ============================================================

init_state("selected_dataset", dataset_options[0])
init_state("selected_layer", None)

init_state("only_saved", True)
init_state("min_purity", 0.0)
init_state("min_mean_activation", 0.0)
init_state(
    "min_valid_top_count",
    min(5, TOP_K_FOR_PURITY),
)
init_state("majority_label_filter", "")
init_state("selection_mode", "Ranked list")
init_state(
    "ranking_method",
    "Sparse class-specific score",
)
init_state("candidate_list_size", 100)
init_state("top_k_display", 10)
init_state("cols_per_row", 5)


# ============================================================
# Sidebar: dataset and layer
# ============================================================

st.sidebar.header("Controls")

keep_valid_selection(
    "selected_dataset",
    dataset_options,
)

dataset_name = st.sidebar.selectbox(
    "Dataset",
    dataset_options,
    index=dataset_options.index(
        st.session_state.selected_dataset
    ),
    key="selected_dataset",
)


available_layers = available_datasets[dataset_name]
layer_options = list(available_layers.keys())

keep_valid_selection(
    "selected_layer",
    layer_options,
)

layer_name = st.sidebar.selectbox(
    "Layer",
    layer_options,
    index=layer_options.index(
        st.session_state.selected_layer
    ),
    key="selected_layer",
)

feature_dir = available_layers[layer_name]


# ============================================================
# Load selected dataset/layer
# ============================================================

try:
    data = load_layer_data(
        dataset_name=dataset_name,
        layer_name=layer_name,
        feature_dir_str=str(feature_dir),
        top_k=TOP_K_FOR_PURITY,
    )
except Exception as error:
    st.error(str(error))
    st.info(
        "Run compute_feature_stats.py for this "
        "dataset/layer before opening it in the app."
    )
    st.stop()


summary = data["summary"].copy()
num_features = len(summary)

st.sidebar.write("Feature directory:")
st.sidebar.code(str(feature_dir))

st.sidebar.write(
    f"Number of features: **{num_features}**"
)


# ============================================================
# Sidebar: filtering
# ============================================================

st.sidebar.subheader("Filtering")

only_saved = st.sidebar.checkbox(
    "Only features with saved image folders",
    key="only_saved",
)

min_purity = st.sidebar.slider(
    "Minimum label purity",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    key="min_purity",
)

min_mean_activation = st.sidebar.number_input(
    "Minimum mean activation",
    step=0.1,
    key="min_mean_activation",
)

min_valid_top_count = st.sidebar.slider(
    "Minimum valid top images",
    min_value=1,
    max_value=TOP_K_FOR_PURITY,
    key="min_valid_top_count",
)

majority_label_filter = st.sidebar.text_input(
    "Majority label filter, optional",
    key="majority_label_filter",
    help=(
        "Enter a numeric label ID. "
        "Leave empty to include every label."
    ),
)


filtered = summary.copy()

if only_saved:
    filtered = filtered[
        filtered["has_saved_folder"] == True
    ]

filtered = filtered[
    filtered["label_purity"] >= min_purity
]

filtered = filtered[
    filtered["mean_activation"]
    >= min_mean_activation
]

if "valid_top_count" in filtered.columns:
    filtered = filtered[
        filtered["valid_top_count"]
        >= min_valid_top_count
    ]

if majority_label_filter.strip():
    try:
        label_id = int(
            majority_label_filter.strip()
        )

        filtered = filtered[
            filtered["majority_label"]
            == label_id
        ]

    except ValueError:
        st.sidebar.warning(
            "Majority label must be an integer."
        )


# ============================================================
# Sidebar: feature selection
# ============================================================

st.sidebar.subheader("Feature selection")

selection_mode_options = [
    "Ranked list",
    "Manual feature ID",
]

keep_valid_selection(
    "selection_mode",
    selection_mode_options,
)

selection_mode = st.sidebar.radio(
    "Selection mode",
    selection_mode_options,
    index=selection_mode_options.index(
        st.session_state.selection_mode
    ),
    key="selection_mode",
)


selected_feature_key = feature_state_key(
    dataset_name,
    layer_name,
)

manual_feature_key = manual_feature_state_key(
    dataset_name,
    layer_name,
)

init_state(selected_feature_key, 0)
init_state(manual_feature_key, 0)


if selection_mode == "Manual feature ID":
    current_manual_value = int(
        st.session_state[manual_feature_key]
    )

    current_manual_value = max(
        0,
        min(current_manual_value, num_features - 1),
    )

    st.session_state[manual_feature_key] = (
        current_manual_value
    )

    feature_id = st.sidebar.number_input(
        "Feature ID",
        min_value=0,
        max_value=num_features - 1,
        step=1,
        key=manual_feature_key,
    )

    feature_id = int(feature_id)

    st.session_state[selected_feature_key] = (
        feature_id
    )


else:
    ranking_method_options = [
        "Mean activation",
        "Sparsity",
        "Label purity",
        "Class-specific score",
        "Sparse class-specific score",
        "Saved image folders only",
    ]

    keep_valid_selection(
        "ranking_method",
        ranking_method_options,
    )

    ranking_method = st.sidebar.selectbox(
        "Ranking method",
        ranking_method_options,
        index=ranking_method_options.index(
            st.session_state.ranking_method
        ),
        key="ranking_method",
    )

    top_n = st.sidebar.slider(
        "Candidate list size",
        min_value=10,
        max_value=500,
        step=10,
        key="candidate_list_size",
    )

    ranked_df = select_feature_by_ranking(
        summary=filtered,
        ranking_method=ranking_method,
        top_n=top_n,
    )

    if ranked_df.empty:
        st.warning(
            "No features match the current filters."
        )
        st.stop()

    feature_options = (
        ranked_df["feature_id"]
        .astype(int)
        .tolist()
    )

    keep_valid_selection(
        selected_feature_key,
        feature_options,
    )

    feature_id = st.sidebar.selectbox(
        "Feature ID",
        feature_options,
        index=feature_options.index(
            st.session_state[selected_feature_key]
        ),
        key=selected_feature_key,
    )

    feature_id = int(feature_id)

    st.session_state[manual_feature_key] = (
        feature_id
    )


# ============================================================
# Main area: feature summary
# ============================================================

feature_ids_in_summary = (
    summary["feature_id"]
    .astype(int)
    .values
)

if feature_id not in feature_ids_in_summary:
    st.error(
        f"Feature ID {feature_id} was not found "
        f"for {dataset_name} / {layer_name}."
    )
    st.stop()


feature_row = summary[
    summary["feature_id"] == feature_id
].iloc[0]


st.subheader(
    f"Dataset: {dataset_name} | "
    f"Layer: {layer_name} | "
    f"Feature: {feature_id}"
)


col1, col2, col3, col4, col5, col6 = (
    st.columns(6)
)

col1.metric(
    "Mean activation",
    f"{feature_row['mean_activation']:.4f}",
)

col2.metric(
    "Sparsity",
    f"{feature_row['sparsity']:.6f}",
)

col3.metric(
    f"Purity top-{TOP_K_FOR_PURITY}",
    f"{feature_row['label_purity']:.2f}",
)

col4.metric(
    "Majority label",
    str(int(feature_row["majority_label"])),
)

col5.metric(
    "Unique labels",
    str(int(feature_row["unique_label_count"])),
)

col6.metric(
    "Saved folder",
    (
        "yes"
        if bool(feature_row["has_saved_folder"])
        else "no"
    ),
)


extra_line = (
    f"Top image index: "
    f"**{int(feature_row['top_image_index'])}**, "
    f"top activation: "
    f"**{feature_row['top_activation_value']:.4f}**, "
    f"top label: "
    f"**{int(feature_row['top_label'])}**"
)

if "valid_top_count" in feature_row.index:
    extra_line += (
        f", valid top images: "
        f"**{int(feature_row['valid_top_count'])}**"
    )

st.write(extra_line)


# ============================================================
# Main area: top activation table
# ============================================================

top_k_display = st.slider(
    "Top activating entries to show",
    min_value=1,
    max_value=min(
        30,
        int(data["values"].shape[1]),
    ),
    key="top_k_display",
)


top_table = get_top_activation_table(
    data=data,
    feature_id=feature_id,
    top_k_display=top_k_display,
)

st.markdown("### Top activating entries")

try:
    st.dataframe(
        top_table,
        use_container_width=True,
    )
except TypeError:
    st.dataframe(top_table)


# ============================================================
# Main area: image grid
# ============================================================

st.markdown("### Top activating images")

image_paths = get_feature_image_paths(
    feature_dir=data["feature_dir"],
    feature_id=feature_id,
)

st.write(
    f"Image files found for this feature: "
    f"**{len(image_paths)}**"
)


if not image_paths:
    st.warning(
        "No saved image folder was found for this "
        "feature. The activation table is still available."
    )

else:
    image_paths = image_paths[:top_k_display]

    cols_per_row = st.slider(
        "Images per row",
        min_value=2,
        max_value=8,
        key="cols_per_row",
    )

    for start in range(
        0,
        len(image_paths),
        cols_per_row,
    ):
        columns = st.columns(cols_per_row)

        row_images = image_paths[
            start:start + cols_per_row
        ]

        for column, image_path in zip(
            columns,
            row_images,
        ):
            metadata = parse_image_filename(
                image_path
            )

            with column:
                try:
                    image = Image.open(
                        image_path
                    ).convert("RGB")

                    if metadata is not None:
                        caption = (
                            f"rank={metadata['rank']} | "
                            f"idx={metadata['image_index']} | "
                            f"act={metadata['activation']:.3f} | "
                            f"label={metadata['label']}"
                        )
                    else:
                        caption = image_path.name

                    show_image(
                        image=image,
                        caption=caption,
                    )

                except Exception as error:
                    st.error(
                        "Could not open image: "
                        f"{image_path.name}"
                    )
                    st.caption(str(error))


# ============================================================
# Main area: filtered table
# ============================================================

with st.expander("Show filtered feature table"):
    preview_columns = [
        "feature_id",
        "label_purity",
        "majority_label",
        "unique_label_count",
        "mean_activation",
        "sparsity",
        "has_saved_folder",
        "class_specific_score",
    ]

    if "valid_top_count" in filtered.columns:
        preview_columns.insert(
            4,
            "valid_top_count",
        )

    available_preview_columns = [
        column
        for column in preview_columns
        if column in filtered.columns
    ]

    table_to_display = filtered[
        available_preview_columns
    ].copy()

    if "class_specific_score" in table_to_display.columns:
        table_to_display = (
            table_to_display
            .sort_values(
                "class_specific_score",
                ascending=False,
            )
        )

    table_to_display = table_to_display.head(500)

    try:
        st.dataframe(
            table_to_display,
            use_container_width=True,
        )
    except TypeError:
        st.dataframe(table_to_display)
