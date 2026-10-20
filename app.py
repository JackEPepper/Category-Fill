"""Drag-and-drop app: streamlit run app.py"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
from category_fill import explain
from category_fill.analysis import value_rules
from category_fill.data import is_categorical
from category_fill.filters import Rule, fmt_num, population_mask, rules_from_json, rules_summary, rules_to_json
from category_fill.report import summary_frame
from category_fill.sample import make_loans

BAR = "#2a78d6"
# Last uploaded file is kept here (on this computer only) so a page refresh or expired session can reopen it.
CACHE_DIR = Path(__file__).parent / ".category_fill_cache"
st.set_page_config(page_title="Category Fill", layout="wide")


def _last_file() -> tuple[str, bytes, str] | None:
    try:
        meta = json.loads((CACHE_DIR / "last.json").read_text())
        return meta["name"], (CACHE_DIR / "last.bin").read_bytes(), meta["saved"]
    except (OSError, ValueError, KeyError):
        return None


def _remember(upload) -> None:
    try:
        meta_path = CACHE_DIR / "last.json"
        if meta_path.exists() and json.loads(meta_path.read_text()).get("file_id") == upload.file_id:
            return
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / "last.bin").write_bytes(upload.getvalue())
        saved = datetime.now().strftime("%Y-%m-%d %H:%M")
        meta_path.write_text(json.dumps({"name": upload.name, "saved": saved, "file_id": upload.file_id}))
    except OSError:
        pass


def _forget() -> None:
    for f in ("last.json", "last.bin"):
        (CACHE_DIR / f).unlink(missing_ok=True)


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


@st.cache_data(show_spinner="Looking for values that limit the category")
def _value_rules(df: pd.DataFrame, target: str, cands: tuple) -> pd.DataFrame:
    return value_rules(df, target, list(cands))


# Results are kept on the server, so re-running the same model after a refresh is instant.
@st.cache_resource(show_spinner=False, max_entries=6)
def _train(df: pd.DataFrame, target: str, features: tuple, cfg: TrainConfig, rules: tuple):
    """``rules`` is a tuple of (feature, value, allowed classes) so it can be part of the cache key."""
    frame = pd.DataFrame([{"feature": f, "value": v, "classes": list(c)} for f, v, c in rules])
    return train_and_fill(df, target, list(features), cfg, value_rules=frame)


def _surrogate(df: pd.DataFrame, res, depth: int, focus: str | None):
    cache = st.session_state.setdefault("surrogates", {})
    key = (id(res), depth, focus)
    if key not in cache:
        cache[key] = explain.fit_surrogate(df, res, max_depth=depth, focus=focus)
    return cache[key]


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


def _new_rule(rule: Rule | None = None) -> None:
    """Register a rule id; when loading a saved rule, pre-seed its widget state."""
    ss = st.session_state
    rid = ss.next_rule
    ss.next_rule += 1
    ss.rule_ids.append(rid)
    if rule is not None:
        ss[f"r{rid}_mode"] = "Exclude" if rule.mode == "exclude" else "Keep only"
        ss[f"r{rid}_col"] = rule.column
        ss[f"r{rid}_vals_{rule.column}"] = list(rule.values)
        ss[f"r{rid}_min_{rule.column}"] = rule.min
        ss[f"r{rid}_max_{rule.column}"] = rule.max
        ss[f"r{rid}_blank_{rule.column}"] = rule.blank


def population_editor(frame: pd.DataFrame) -> list[Rule]:
    ss = st.session_state
    ss.setdefault("rule_ids", [])
    ss.setdefault("next_rule", 0)
    columns = list(frame.columns)

    st.markdown(
        "Take loans out of the analysis before anything else runs. Each rule looks at **one column**. "
        "**Exclude** drops the loans it matches, and **Keep only** drops the loans it doesn't match. "
        "Add one rule per column, and a loan stays in only if it passes every rule. Excluded loans are "
        "not used for training and are not filled. They stay in the download, marked `excluded`."
    )

    saved = st.file_uploader("Load saved rules (.json)", type="json", key="rules_upload")
    if saved is not None and ss.get("rules_loaded") != saved.file_id:
        try:
            loaded = rules_from_json(saved.getvalue().decode())
        except (ValueError, TypeError) as e:
            st.error(f"Couldn't read rules file: {e}")
        else:
            missing = [r.column for r in loaded if r.column not in columns]
            if missing:
                st.warning("Skipped rules for columns not in this file: " + ", ".join(missing))
            ss.rule_ids = []
            for r in loaded:
                if r.column in columns:
                    _new_rule(r)
        ss.rules_loaded = saved.file_id

    rules = []
    for rid in list(ss.rule_ids):
        with st.container(border=True):
            c_mode, c_col, c_del = st.columns([2, 3, 1], vertical_alignment="bottom")
            mode = c_mode.radio("Action", ["Exclude", "Keep only"], key=f"r{rid}_mode", horizontal=True)
            ckey = f"r{rid}_col"
            col = c_col.selectbox(
                "Column", columns, key=ckey, placeholder="Pick a column", **({} if ckey in ss else {"index": None})
            )
            if c_del.button("Remove", key=f"r{rid}_del"):
                ss.rule_ids.remove(rid)
                st.rerun()
            if col is None:
                continue

            s = frame[col]
            values, lo, hi = [], None, None
            if is_categorical(s) or s.nunique() <= 30:
                counts = s.value_counts()
                options = list(counts.index[:2000])
                vkey = f"r{rid}_vals_{col}"
                if vkey in ss:  # map saved values onto this file's options, dropping ones that don't exist
                    by_text = {str(o): o for o in options}
                    ss[vkey] = [by_text[str(v)] for v in ss[vkey] if str(v) in by_text]
                values = st.multiselect(
                    f"Values of {col}" + (" (top 2,000 shown)" if len(counts) > 2000 else ""),
                    options,
                    key=vkey,
                    format_func=lambda v, c=counts: f"{v:g}  ({c[v]:,})" if isinstance(v, float) else f"{v}  ({c[v]:,})",
                )
            else:
                c_lo, c_hi = st.columns(2)
                kmin, kmax = f"r{rid}_min_{col}", f"r{rid}_max_{col}"
                lo = c_lo.number_input(f"{col} from", key=kmin, **({} if kmin in ss else {"value": None}))
                hi = c_hi.number_input(f"{col} to", key=kmax, **({} if kmax in ss else {"value": None}))
                st.caption(f"Range in data: {fmt_num(s.min())} to {fmt_num(s.max())}. Leave an end empty for no limit.")
            blank = st.checkbox(f"Match loans where {col} is blank ({int(s.isna().sum()):,})", key=f"r{rid}_blank_{col}")

            rule = Rule(
                column=col,
                values=[v.item() if hasattr(v, "item") else v for v in values],
                min=lo,
                max=hi,
                blank=blank,
                mode="exclude" if mode == "Exclude" else "keep",
            )
            rules.append(rule)
            if rule.is_empty:
                st.caption("No criteria yet, so this rule has no effect.")
            else:
                st.caption(f"{rule.describe()}. Matches {int(rule.matches(frame).sum()):,} loans.")

    if st.button("+ Add rule"):
        _new_rule()
        st.rerun()

    active = [r for r in rules if not r.is_empty]
    if active:
        keep = population_mask(frame, active)
        c1, c2 = st.columns(2)
        c1.metric("Loans in population", f"{int(keep.sum()):,}", f"of {len(frame):,}", delta_color="off", delta_arrow="off")
        c2.metric("Loans excluded", f"{int((~keep).sum()):,}")
        st.dataframe(rules_summary(frame, active), hide_index=True, width="stretch")
        st.download_button("Save these rules (.json)", rules_to_json(active), "population_rules.json", "application/json")
    return active


# ---------------------------------------------------------------- sidebar: data in
st.sidebar.title("Category Fill")
upload = st.sidebar.file_uploader("Drop your loan file here", type=["csv", "tsv", "xlsx", "xls", "parquet"])
last = _last_file()
reopen = False
if upload is not None:
    if st.sidebar.checkbox(
        "Remember this file on this computer",
        value=True,
        help=f"Keeps a copy in {CACHE_DIR} so a refresh or expired session can reopen it. Untick to delete it.",
    ):
        _remember(upload)
    else:
        _forget()
elif last is not None:
    reopen = st.sidebar.toggle(f"Reopen last file: {last[0]}", value=True, help=f"Saved {last[2]}")
use_sample = st.sidebar.toggle("Use synthetic sample data instead", value=upload is None and not reopen)

if upload is not None and not use_sample:
    raw = _load(upload.getvalue(), upload.name)
    source_name = upload.name.rsplit(".", 1)[0]
elif reopen and not use_sample:
    raw = _load(last[1], last[0])
    source_name = last[0].rsplit(".", 1)[0]
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
df_all = _clean(raw, tokens, tuple(force_cat))

tab_pop, tab_data, tab_feat, tab_model, tab_pred, tab_explain = st.tabs(
    ["0 · Population", "1 · Data", "2 · Features", "3 · Model readout", "4 · Filled data", "5 · Explain"]
)

# ---------------------------------------------------------------- 0. population
with tab_pop:
    rules = population_editor(df_all)
in_pop = population_mask(df_all, rules)
df = df_all.loc[in_pop]
rules_key = rules_to_json(rules)
st.sidebar.metric(
    "Loans in population", f"{len(df):,}", f"{len(df_all) - len(df):,} excluded" if rules else "no exclusions",
    delta_color="off", delta_arrow="off",
)
if df.empty:
    st.error("Your population rules exclude every loan. Loosen them in tab 0.")
    st.stop()

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
    depth = st.slider("XGBoost tree depth", 2, 12, 6)
    lr = st.select_slider("Learning rate", [0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)
cfg = TrainConfig(test_size=test_size, min_class_count=int(min_class), balance_classes=balance, max_depth=depth, learning_rate=lr)

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

    st.subheader("Must-use features and values that limit the category")
    st.markdown(
        "Some values narrow the answer down on their own, for example a yes/no indicator where **Y always "
        "means one of a family of categories** (like a BCC indicator that always means one of the BCC "
        "templates), or exactly one category. These are found automatically below. A value qualifies when "
        "it is seen on 20+ labelled loans and 98%+ of them fall in a small set of categories. Each ticked "
        "rule makes its column a **must-use feature**: it can't be dropped by screening, and nearly every "
        "tree in the model is allowed to use it."
    )
    found = _value_rules(df, target, candidates)
    if len(found):
        edited = st.data_editor(
            found.assign(use=True),
            hide_index=True,
            width="stretch",
            disabled=[c for c in found.columns],
            column_config={
                "use": st.column_config.CheckboxColumn("use", help="Untick to ignore this rule"),
                "classes": st.column_config.ListColumn("allowed categories", width="large"),
                "n_classes": st.column_config.NumberColumn("# allowed"),
                "coverage": st.column_config.NumberColumn(
                    "coverage", format="percent", help="Share of labelled loans with this value that are in the allowed set"
                ),
                "value": st.column_config.TextColumn("value"),
            },
            key=f"value_rules_{target}",
        )
        chosen_rules = edited.loc[edited["use"]].drop(columns="use")
    else:
        st.caption("No limiting values found automatically in the selected candidate columns.")
        chosen_rules = found

    manual_key = f"manual_rules_{target}"
    manual = st.session_state.setdefault(manual_key, [])
    with st.expander(f"Add a rule by hand ({len(manual)} added)"):
        st.caption("For a family you know about, e.g. *bcc_indicator = Y → any of the BCC templates*.")
        c1, c2 = st.columns(2)
        m_col = c1.selectbox("Column", list(candidates), index=None, placeholder="Pick a column", key="m_col")
        m_val = None
        if m_col is not None:
            vals = df[m_col].dropna().value_counts().index[:500]
            m_val = c2.selectbox("Value", list(vals), index=None, placeholder="Pick a value", key=f"m_val_{m_col}")
        all_classes = sorted(df[target].dropna().astype(str).unique())
        m_classes = st.multiselect("Allowed categories", all_classes, key="m_classes")
        if st.button("Add rule", disabled=m_col is None or m_val is None or not m_classes):
            num = None if is_categorical(df[m_col]) else float(m_val)
            shown = str(m_val) if num is None else (str(int(num)) if num.is_integer() else f"{num:g}")
            manual.append({"feature": m_col, "value": shown, "classes": list(m_classes)})
            st.rerun()
        for i, r in enumerate(list(manual)):
            a, b = st.columns([5, 1])
            a.write(f"**{r['feature']} = {r['value']}** → {' | '.join(r['classes'])}")
            if b.button("Remove", key=f"m_del_{i}"):
                manual.pop(i)
                st.rerun()
    rule_rows = list(chosen_rules[["feature", "value", "classes"]].to_dict("records")) + manual
    rules_for_model = tuple((r["feature"], str(r["value"]), tuple(r["classes"])) for r in rule_rows)

    required = st.multiselect(
        "Always include these features",
        list(candidates),
        default=sorted({r[0] for r in rules_for_model}),
        help="Added to the model even if not selected above, and favoured when trees sample columns.",
    )
    apply_rules = st.toggle(
        "Limit predictions to the categories each ticked rule allows",
        value=True,
        help="A loan with a rule's value can only be given one of that rule's allowed categories. The model "
        "still decides which one. Applies to the test-set readout too, so the accuracy you see is what you get.",
    )
    features = list(dict.fromkeys(features + required))
    cfg = replace(cfg, required_features=tuple(required), apply_value_rules=apply_rules)

# ---------------------------------------------------------------- 3. model
with tab_model:
    run = st.button("Train model", type="primary", disabled=not features)
    if run:
        try:
            with st.spinner("Training XGBoost and scoring the test set"):
                res = _train(df, target, tuple(features), cfg, rules_for_model)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        st.session_state.update(result=res, curve=None, result_key=(source_name, target, rules_key))

    res = st.session_state.get("result")
    if res is None or st.session_state.get("result_key") != (source_name, target, rules_key):
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

        if len(res.rule_check):
            st.subheader("Do the rules hold, and does the model respect them?")
            st.dataframe(
                res.rule_check,
                hide_index=True,
                width="stretch",
                column_config={
                    "actual_in_allowed": st.column_config.NumberColumn(
                        "actual in allowed (test)", format="percent", help="How well the rule itself holds on held-out loans"
                    ),
                    "model_in_allowed_test": st.column_config.NumberColumn(
                        "model in allowed (test)", format="percent", help="Before limiting: the model on its own"
                    ),
                    "model_in_allowed_blank": st.column_config.NumberColumn(
                        "model in allowed (blank rows)", format="percent", help="Before limiting: the model on its own"
                    ),
                },
            )
            weak = res.rule_check[res.rule_check["actual_in_allowed"] < 0.95]
            if len(weak):
                st.warning(
                    "These rules don't hold on held-out loans (under 95% of actual categories are in the allowed "
                    "set). Untick or remove them in tab 2: "
                    + "; ".join(f"**{r.rule}** ({r.actual_in_allowed:.0%})" for r in weak.itertuples())
                )
            if res.config.apply_value_rules:
                st.caption(
                    f"Predictions are limited to each rule's allowed categories. Test accuracy with limits "
                    f"{m['accuracy']:.1%}, without {m['accuracy_without_rules']:.1%}. In tab 4, `set_by` shows "
                    "where a rule changed the model's pick and `rules_applied` lists the rules on each row."
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
        shown = cm.div(cm.sum(axis=1).replace(0, 1), axis=0) if norm else cm
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
    if res is None or st.session_state.get("result_key") != (source_name, target, rules_key):
        st.info("Train a model first (tab 3).")
    else:
        min_conf = st.slider(
            "Only fill rows where the model is at least this confident",
            0.0, 0.99, 0.0, 0.05,
            help="Rows below the cut-off stay blank and are marked 'blank (low confidence)'.",
        )
        filled = fill_frame(raw, res, min_conf, in_population=in_pop)
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
            report_workbook(res, filled, assoc, st.session_state.get("curve"), rules_summary(df_all, rules)),
            f"{source_name}_report.xlsx",
        )
        with st.expander("Summary"):
            st.dataframe(summary_frame(res), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5. explain
with tab_explain:
    res = st.session_state.get("result")
    if res is None or st.session_state.get("result_key") != (source_name, target, rules_key):
        st.info("Train a model first (tab 3).")
    else:
        st.markdown(
            "The model itself is hundreds of small trees, which can't be read. This tab draws **one** decision tree "
            "trained to give the same answers as the model. **Agreement** shows how often it does, so you know how "
            "faithful the picture is. A single category against the rest is usually 90%+ faithful in 3 levels. "
            "All categories at once needs more levels."
        )
        c1, c2 = st.columns(2)
        ALL = "All categories at once (less faithful)"
        by_size = list(res.per_class.sort_values("train_rows", ascending=False).index)
        choice = c1.selectbox("Explain", by_size + [ALL], help="One category vs. the rest gives the clearest, most faithful tree.")
        focus = None if choice == ALL else choice
        tree_depth = c2.slider("Levels of questions", 2, 6, 4 if focus is None else 3)
        with st.spinner("Fitting explanation tree"):
            sur = _surrogate(df, res, tree_depth, focus)
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Agreement with the model",
            f"{sur.fidelity:.1%}",
            f"vs {sur.baseline_fidelity:.1%} by always giving the most common answer",
            delta_color="off",
            delta_arrow="off",
            help="On loans the explanation tree didn't see. The comparison shows how much the questions add.",
        )
        m2.metric(
            "Explanation tree right vs. actual",
            f"{sur.accuracy:.1%}" if sur.accuracy == sur.accuracy else "n/a",
            help="Share of labelled loans (not used to build the tree) where the tree's answer matches the real category.",
        )
        m3.metric(
            "Full model accuracy (test set)",
            f"{res.metrics['accuracy']:.1%}",
            help="From tab 3. Measured on held-out loans the model never trained on." if focus is None
            else "From tab 3, across all categories. Measured on held-out loans the model never trained on.",
        )
        if sur.fidelity < 0.8:
            st.warning(
                "This simplified tree only matches the model on "
                f"{sur.fidelity:.0%} of loans. Add levels, or explain one category at a time for a more faithful picture."
            )
        fig = explain.tree_figure(sur)
        st.pyplot(fig, width="stretch")
        stem = f"{source_name}_{target}_tree" + (f"_{focus}" if focus else "") + f"_depth{tree_depth}"
        stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
        d1, d2, d3, d4 = st.columns(4)
        d1.download_button("Download PNG", explain.figure_bytes(fig, "png"), f"{stem}.png", "image/png")
        d2.download_button("Download SVG", explain.figure_bytes(fig, "svg"), f"{stem}.svg", "image/svg+xml")
        d3.download_button("Download PDF", explain.figure_bytes(fig, "pdf"), f"{stem}.pdf", "application/pdf")
        d4.download_button("Download rules (.txt)", explain.rules_text(sur), f"{stem}_rules.txt", "text/plain")
        plt.close(fig)
        st.subheader("The same tree as IF/THEN rules")
        st.dataframe(
            explain.rules_frame(sur),
            hide_index=True,
            width="stretch",
            column_config={"share": st.column_config.NumberColumn("share of loans in group", format="percent")},
        )
