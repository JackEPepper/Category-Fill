"""Drag-and-drop app: streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from category_fill import (
    TrainConfig,
    class_distribution,
    clean_frame,
    feature_associations,
    feature_subset_curve,
    fill_frame,
    load_table,
    missingness_shift,
    profile_columns,
    report_workbook,
    suggest_targets,
    train_and_fill,
)
from category_fill.report import summary_frame
from category_fill.sample import make_loans

BAR = "#2a78d6"
st.set_page_config(page_title="Category Fill", layout="wide")


@st.cache_data(show_spinner="Reading file")
def _load(data: bytes, name: str) -> pd.DataFrame:
    return load_table(data, filename=name)


@st.cache_data(show_spinner="Generating sample loans")
def _sample() -> pd.DataFrame:
    return make_loans()


@st.cache_data(show_spinner="Cleaning columns")
def _clean(raw: pd.DataFrame, tokens: tuple, force_cat: tuple) -> pd.DataFrame:
    return clean_frame(raw, extra_missing_tokens=tokens, force_categorical=force_cat)


@st.cache_data(show_spinner="Screening features")
def _assoc(df: pd.DataFrame, target: str, cands: tuple) -> pd.DataFrame:
    return feature_associations(df, target, list(cands))


def hbar(frame: pd.DataFrame, x: str, y: str, title: str, fmt: str = ".2f"):
    fig = px.bar(frame, x=x, y=y, orientation="h", title=title, text_auto=fmt)
    fig.update_traces(marker_color=BAR, marker_line_width=0, textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "title": None},
        xaxis={"title": None, "showgrid": True, "gridcolor": "rgba(128,128,128,0.15)"},
        height=max(260, 26 * len(frame) + 90),
        margin={"l": 10, "r": 40, "t": 50, "b": 10},
        bargap=0.25,
    )
    return fig


# ---------------------------------------------------------------- sidebar: data in
st.sidebar.title("Category Fill")
upload = st.sidebar.file_uploader("Drop your loan file here", type=["csv", "tsv", "xlsx", "xls", "parquet"])
use_sample = st.sidebar.toggle("Use synthetic sample data instead", value=upload is None)

if upload is not None and not use_sample:
    raw = _load(upload.getvalue(), upload.name)
    source_name = upload.name.rsplit(".", 1)[0]
elif use_sample:
    raw = _sample()
    source_name = "sample_loans"
else:
    st.info("Drop a CSV or Excel file in the sidebar to begin.")
    st.stop()

with st.sidebar.expander("Data cleaning options"):
    tokens = st.text_input(
        "Extra values that mean blank (comma-separated)",
        help="Blank, NA, N/A, null, none, #N/A, '-' and '?' are already treated as missing. "
        "Add things like 'Unknown' or 'TBD' if your system uses them for the missing category.",
    )
    force_cat = st.multiselect(
        "Numeric-looking columns to treat as categories",
        options=list(raw.columns),
        help="E.g. branch numbers or product codes where 12 isn't 'bigger' than 11.",
    )
tokens = tuple(t.strip() for t in tokens.split(",") if t.strip())
df = _clean(raw, tokens, tuple(force_cat))

suggested = suggest_targets(df)
options = list(df.columns)
target = st.sidebar.selectbox(
    "Column to fill",
    options,
    index=options.index(suggested[0]) if suggested else 0,
    help="Auto-selected: the categorical column whose missing share is closest to ~15%.",
)
n_blank = int(df[target].isna().sum())
st.sidebar.metric("Blank rows to fill", f"{n_blank:,}", f"{n_blank / len(df):.1%} of {len(df):,}", delta_color="off", delta_arrow="off")

excluded = st.sidebar.multiselect("Never use as features", [c for c in options if c != target])
candidates = tuple(c for c in options if c != target and c not in excluded)
assoc = _assoc(df, target, candidates)

with st.sidebar.expander("Model settings"):
    test_size = st.slider("Test set share", 0.1, 0.4, 0.2, 0.05, help="Labelled rows held back to grade the model.")
    min_class = st.number_input("Min rows per class", 2, 500, 10, help="Rarer classes can't be predicted reliably.")
    balance = st.checkbox("Balance classes", help="Better recall on rare classes, usually lower overall accuracy.")
    depth = st.slider("Tree depth", 2, 12, 6)
    lr = st.select_slider("Learning rate", [0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)
cfg = TrainConfig(test_size=test_size, min_class_count=int(min_class), balance_classes=balance, max_depth=depth, learning_rate=lr)

tab_data, tab_feat, tab_model, tab_pred = st.tabs(["1 · Data", "2 · Features", "3 · Model readout", "4 · Filled data"])

# ---------------------------------------------------------------- 1. data
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns (after date expansion)", df.shape[1])
    c3.metric(f"'{target}' classes", int(df[target].nunique()))
    c4.metric("Blank in target", f"{n_blank:,}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Column profile")
        st.dataframe(profile_columns(df), hide_index=True, width="stretch")
    with right:
        dist = class_distribution(df, target)
        st.subheader(f"'{target}' distribution (labelled rows)")
        st.plotly_chart(hbar(dist, "count", "class", "", ",d"), width="stretch")
    with st.expander("Raw data preview"):
        st.dataframe(raw.head(200), width="stretch")

# ---------------------------------------------------------------- 2. features
with tab_feat:
    st.markdown(
        "**Association** is Cramér's V between each column and the target on labelled rows "
        "(0 = no relationship, 1 = column fully determines the target). Columns are **recommended** unless "
        "they look like IDs, are constant, show no signal, or are blank on the rows you need to fill."
    )
    left, right = st.columns([2, 3])
    with left:
        st.plotly_chart(hbar(assoc.head(25), "association", "feature", "Association with target"), width="stretch")
    with right:
        st.dataframe(
            assoc,
            hide_index=True,
            width="stretch",
            column_config={
                "association": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "missing_pct_unlabeled": st.column_config.NumberColumn("missing % on blank-target rows"),
            },
        )

    shift = missingness_shift(df, target, list(candidates))
    big = shift[shift["shift"] > 0.1]
    if len(big):
        st.warning(
            "The rows missing the target look different from the labelled rows on: "
            + ", ".join(f"**{f}** ({v:.2f})" for f, v in zip(big["feature"], big["shift"]))
            + ". Blanks aren't random, so real-world accuracy on them may differ from the test score."
        )
    with st.expander("Are the blank rows different from the labelled rows?"):
        st.dataframe(shift, hide_index=True, width="stretch")

    st.subheader("Features to train on")
    features = st.multiselect(
        "Selected features (pre-filled with the recommended set)",
        list(candidates),
        default=list(assoc.loc[assoc["recommended"], "feature"]),
    )

# ---------------------------------------------------------------- 3. model
with tab_model:
    run = st.button("Train model", type="primary", disabled=not features)
    if run:
        status = st.status("Training", expanded=False)
        try:
            res = train_and_fill(df, target, features, cfg, progress=status.write)
        except ValueError as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            st.stop()
        status.update(label="Done", state="complete")
        st.session_state.update(result=res, curve=None, result_key=(source_name, target))

    res = st.session_state.get("result")
    if res is None or st.session_state.get("result_key") != (source_name, target):
        st.info("Pick features in tab 2, then press **Train model**.")
    else:
        m = res.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{m['accuracy']:.1%}", f"{m['accuracy'] - m['baseline_accuracy']:+.1%} vs baseline")
        c2.metric("Baseline (always guess most common)", f"{m['baseline_accuracy']:.1%}", m["baseline_class"], delta_color="off", delta_arrow="off")
        c3.metric("Right answer in top 3", f"{m['top3_accuracy']:.1%}")
        c4.metric("Macro F1", f"{m['macro_f1']:.3f}", help="Average F1 with every class weighted equally.")
        st.caption(
            f"Split: {m['n_train']:,} train · {m['n_val']:,} validation (early stopping) · {m['n_test']:,} test. "
            f"{m['trees']} trees. Final model refit on all {m['n_labelled']:,} labelled rows."
        )
        if res.dropped_classes:
            st.warning(
                "Too rare to predict (below min rows per class): "
                + ", ".join(f"{k} ({v})" for k, v in res.dropped_classes.items())
            )

        st.subheader("How much to trust each prediction")
        left, right = st.columns(2)
        with left:
            st.markdown("Accuracy on the test set at each confidence level")
            st.dataframe(
                res.confidence_bins,
                hide_index=True,
                width="stretch",
                column_config={"accuracy": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")},
            )
        with right:
            st.markdown("If you only accept predictions above a cut-off")
            st.dataframe(
                res.thresholds,
                hide_index=True,
                width="stretch",
                column_config={
                    "test_coverage": st.column_config.NumberColumn(format="percent"),
                    "test_accuracy": st.column_config.NumberColumn(format="percent"),
                },
            )

        st.subheader("Per-category readout (test set)")
        st.dataframe(
            res.per_class,
            width="stretch",
            column_config={
                c: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")
                for c in ("precision", "recall", "f1")
            },
        )
        st.caption("Precision: when it predicts this class, how often it's right. Recall: of the true cases, how many it finds.")

        cm = res.confusion
        norm = st.toggle("Show confusion matrix as row %", value=True)
        shown = cm.div(cm.sum(1).replace(0, 1), axis=0) if norm else cm
        fig = px.imshow(
            shown,
            text_auto=".0%" if norm else "d",
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "share" if norm else "rows"},
            aspect="auto",
        )
        fig.update_layout(height=max(380, 34 * len(cm) + 120), margin={"t": 30})
        st.plotly_chart(fig, width="stretch")

        st.subheader("Which columns actually matter")
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(
                hbar(res.importance, "accuracy_drop_if_shuffled", "feature", "Accuracy lost when the column is scrambled", ".1%"),
                width="stretch",
            )
        with right:
            st.dataframe(res.importance, hide_index=True, width="stretch")

        st.markdown("**Smallest feature set that works** — retrains with the top 1, 2, 3 … columns from the chart above.")
        if st.button("Run feature-subset test"):
            with st.spinner("Retraining on feature subsets"):
                st.session_state["curve"] = feature_subset_curve(df, target, list(res.importance["feature"]), cfg)
        curve = st.session_state.get("curve")
        if curve is not None:
            fig = px.line(curve, x="n_features", y=["accuracy", "macro_f1"], markers=True)
            fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Number of top features", yaxis_title=None, legend_title=None)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(curve, hide_index=True, width="stretch")

        with st.expander("Test-set predictions (row-level)"):
            only_wrong = st.checkbox("Only show mistakes")
            tp = res.test_predictions
            st.dataframe(tp[~tp["correct"]] if only_wrong else tp, hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4. predictions
with tab_pred:
    res = st.session_state.get("result")
    if res is None or st.session_state.get("result_key") != (source_name, target):
        st.info("Train a model first (tab 3).")
    else:
        min_conf = st.slider(
            "Only fill rows where the model is at least this confident",
            0.0, 0.99, 0.0, 0.05,
            help="Rows below the cut-off stay blank and are marked 'blank (low confidence)'.",
        )
        filled = fill_frame(raw, res, min_conf)
        accepted = int((filled[f"{target}_source"] == "predicted").sum())
        st.metric("Rows filled", f"{accepted:,} of {len(res.predictions):,}")

        preds = res.predictions.merge(raw, left_on="row", right_index=True, how="left")
        st.dataframe(preds[preds["confidence"] >= min_conf], hide_index=True, width="stretch")

        pred_dist = res.predictions.loc[res.predictions["confidence"] >= min_conf, "predicted"].value_counts().reset_index()
        pred_dist.columns = ["class", "count"]
        st.plotly_chart(hbar(pred_dist, "count", "class", "Predicted categories for blank rows", ",d"), width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "Download filled data (CSV)", filled.to_csv(index=False).encode(), f"{source_name}_filled.csv", "text/csv"
        )
        c2.download_button(
            "Download predictions only (CSV)",
            preds.to_csv(index=False).encode(),
            f"{source_name}_predictions.csv",
            "text/csv",
        )
        c3.download_button(
            "Download full report (Excel)",
            report_workbook(res, filled, assoc, st.session_state.get("curve")),
            f"{source_name}_report.xlsx",
        )
        with st.expander("Summary"):
            st.dataframe(summary_frame(res), hide_index=True, width="stretch")
