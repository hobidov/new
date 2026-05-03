import os
from dotenv import load_dotenv
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

try:
    import plotly.express as px
except Exception:
    px = None

PALETTE = {
    "primary": "#4DB7E5",   # soft blue
    "secondary": "#7C3AED", # purple
    "accent": "#14B8A6",    # teal
    "neutral": "#6B7280",   # grey
    "light": "#F8FAFC",     # background
    "gold": "#D4A017"
}

STAGE_COLORS = {
    ("Social", "Spark"): PALETTE["primary"],
    ("Social", "Growth"): PALETTE["secondary"],
    ("Social", "Horizon"): PALETTE["gold"],
    ("Cultural", "Spark"): PALETTE["accent"],
    ("Cultural", "Growth"): PALETTE["secondary"],
    ("Cultural", "Horizon"): PALETTE["primary"],
}

BOT_INTRO = "Ask me about Social Spark, Social Horizon, Cultural Growth, Cultural Horizon, outcomes, or recommendations."
PROMPT_SUGGESTIONS = [
    "What is the strongest outcome area?",
    "Which outcome is weakest?",
    "How are Social and Cultural Horizon performing?",
    "What should Monkey Baa improve next?",
    "Summarize the core findings.",
]

MAPPING_RULES = [
    {"sourceColumn": "It gave me a sense of joy, beauty and wonder slider", "outcome": "Young people experience joy and wonder.", "stage": "Spark", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It meant something to me personally slider", "outcome": "Young people experience a spark of inspiration.", "stage": "Spark", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It is one of the best examples of its type that I have experienced slider", "outcome": "Young people experience a spark of inspiration.", "stage": "Spark", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It inspired my own creativity slider", "outcome": "Young people build confidence and self-esteem.", "stage": "Growth", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It opened my mind to new possibilities slider", "outcome": "Young people demonstrate enhanced empathy and emotional intelligence.", "stage": "Growth", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "It helped me feel part of the community slider", "outcome": "Young people experience greater social inclusion and community connection.", "stage": "Growth", "category": "Social", "normalizer": "zero_to_one"},
    {"sourceColumn": "How likely are you to attend an event/activity by Monkey Baa again? dropdown", "outcome": "Communities experience a lasting increase in social capital and youth engagement.", "stage": "Horizon", "category": "Social", "normalizer": "likelihood"},
    {"sourceColumn": "How likely is it that you would recommend this show to a friend or colleague? dropdown", "outcome": "Communities experience a lasting increase in social capital and youth engagement.", "stage": "Horizon", "category": "Social", "normalizer": "likelihood"},
    {"sourceColumn": "How would you rate your experience overall? dropdown", "outcome": "Young people benefit from improved well-being and create lifelong positive memories.", "stage": "Horizon", "category": "Social", "normalizer": "overall_experience"},
    {"sourceColumn": "The performance was entertaining slider", "outcome": "Young people develop curiosity and engagement with theatre.", "stage": "Spark", "category": "Cultural", "normalizer": "zero_to_one"},
    {"sourceColumn": "The performance was emotionally impactful slider", "outcome": "Young people see themselves in stories and feel validated.", "stage": "Spark", "category": "Cultural", "normalizer": "zero_to_one"},
    {"sourceColumn": "It opened my mind to new possibilities slider", "outcome": "Young people build increased cultural literacy and openness.", "stage": "Growth", "category": "Cultural", "normalizer": "zero_to_one"},
    {"sourceColumn": "How likely is it that you would recommend this show to a friend or colleague? dropdown", "outcome": "Young people develop a growing appreciation for theatre and the arts.", "stage": "Growth", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How likely are you to attend an event/activity by Monkey Baa again? dropdown", "outcome": "Young people and communities become repeat attendees and new audiences are formed.", "stage": "Growth", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How likely is it that you would recommend this show to a friend or colleague? dropdown", "outcome": "Monkey Baa influences the broader arts sector.", "stage": "Horizon", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How likely are you to attend an event/activity by Monkey Baa again? dropdown", "outcome": "A generation of lifelong arts engagers is cultivated.", "stage": "Horizon", "category": "Cultural", "normalizer": "likelihood"},
    {"sourceColumn": "How would you rate your experience overall? dropdown", "outcome": "Australian storytelling is enriched and diversified.", "stage": "Horizon", "category": "Cultural", "normalizer": "overall_experience"},
]

DEFAULT_KPIS = [
    {"label": "Social Spark", "value": 0, "category": "Social", "stage": "Spark", "color": PALETTE["primary"]},
    {"label": "Social Growth", "value": 0, "category": "Social", "stage": "Growth", "color": PALETTE["secondary"]},
    {"label": "Social Horizon", "value": 0, "category": "Social", "stage": "Horizon", "color": PALETTE["gold"]},
    {"label": "Cultural Spark", "value": 0, "category": "Cultural", "stage": "Spark", "color": PALETTE["accent"]},
    {"label": "Cultural Growth", "value": 0, "category": "Cultural", "stage": "Growth", "color": PALETTE["secondary"]},
    {"label": "Cultural Horizon", "value": 0, "category": "Cultural", "stage": "Horizon", "color": PALETTE["primary"]},
]

def safe(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()

def lower(value: Any) -> str:
    return safe(value).lower()

def avg(values: List[float]) -> int:
    return int(round(sum(values) / len(values))) if values else 0

def clamp(n: float, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, int(round(n))))

def normalize_zero_to_one(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)) and not pd.isna(value):
        if 0 <= value <= 1:
            return round(value * 100)
        if 1 < value <= 100:
            return round(value)
        return None
    numeric = pd.to_numeric(safe(value), errors="coerce")
    if pd.isna(numeric):
        return None
    return normalize_zero_to_one(float(numeric))

def normalize_likelihood(value: Any) -> Optional[int]:
    text = lower(value)
    if not text:
        return None
    direct = pd.to_numeric(text, errors="coerce")
    if not pd.isna(direct):
        return clamp((float(direct) / 10) * 100)
    if "extremely likely" in text:
        return 100
    if "very likely" in text:
        return 90
    if text == "likely":
        return 75
    if "neutral" in text:
        return 50
    if "unlikely" in text:
        return 25
    match = re.search(r"\b(10|[1-9])\b", text)
    return clamp((int(match.group(1)) / 10) * 100) if match else None

def normalize_overall_experience(value: Any) -> Optional[int]:
    text = lower(value)
    if not text:
        return None
    if "excellent" in text:
        return 100
    if "good" in text:
        return 75
    if "neutral" in text:
        return 50
    if "poor" in text:
        return 25
    return None

def decode_text(value: str) -> str:
    return (
        value.replace("?üòä", "Happy")
        .replace("�", "Happy")
        .replace("?üßê", "Curious")
        .replace("?üòÆ", "Surprised")
        .replace("?üò®", "Scared")
        .replace("?üòê", "Bored")
        .replace("?üòï", "Confused")
        .replace("‚Äô", "'")
        .strip()
    )

def split_multi_select(value: str) -> List[str]:
    return [
        part.replace("'", "").strip()
        for part in decode_text(value).replace("[", "").replace("]", "").replace("', '", ",").split(",")
        if part.replace("'", "").strip()
    ]

def infer_audience(row: Dict[str, Any]) -> str:
    return safe(row.get("What title best describes you? dropdown")) or safe(row.get("Which category does the respondent belong to? shorttext")) or "Respondent"

def infer_show(row: Dict[str, Any]) -> str:
    return safe(row.get("What Monkey Baa show did you recently attend? dropdown")) or "Unknown show"

def infer_location(row: Dict[str, Any]) -> str:
    return safe(row.get("Where did you see the show? dropdown")) or safe(row.get("Location")) or "Unknown"

def infer_date(rows: List[Dict[str, Any]]) -> str:
    for row in rows:
        for key in row.keys():
            if "date" in key.lower():
                value = str(row.get(key))
                if value:
                    return value
    return "March 2025"


def extract_teacher_quote(rows: List[Dict[str, Any]]) -> str:
    for row in rows:
        for key in row.keys():
            if "comment" in key.lower() or "feedback" in key.lower():
                text = str(row.get(key)).strip()
                if len(text) > 20:
                    return text
    return "Students were highly engaged and continued discussing the performance after the session."

def get_required_columns() -> List[str]:
    return sorted(set(rule["sourceColumn"] for rule in MAPPING_RULES))

def run_normalizer(kind: str, value: Any) -> Optional[int]:
    if kind == "zero_to_one":
        return normalize_zero_to_one(value)
    if kind == "likelihood":
        return normalize_likelihood(value)
    if kind == "overall_experience":
        return normalize_overall_experience(value)
    return None

def normalize_show_name(name: str) -> str:
    name = safe(name).lower()

    if "green sheep" in name:
        return "Where is the Green Sheep"

    if "edward" in name:
        return "Edward the Emu"

    if "josephine" in name:
        return "Josephine Wants To Dance"

    if "possum" in name:
        return "Possum Magic"

    if "peasant" in name:
        return "The Peasant Prince"

    return name.title() if name else "Unknown show"

def map_survey_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for row in rows:
        audience = infer_audience(row)
        show = normalize_show_name(infer_show(row))
        location = infer_location(row)
        for rule in MAPPING_RULES:
            score = run_normalizer(rule["normalizer"], row.get(rule["sourceColumn"]))
            if score is None:
                continue
            mapped.append(
                {
                    "audience": audience,
                    "show": show,
                    "location": location,
                    "question": rule["sourceColumn"],
                    "score": score,
                    "outcome": rule["outcome"],
                    "stage": rule["stage"],
                    "category": rule["category"],
                    "sourceColumn": rule["sourceColumn"],
                }
            )
    return mapped

def build_kpi(label: str, rows: List[Dict[str, Any]], category: str, stage: str, color: str) -> Dict[str, Any]:
    return {"label": label, "value": avg([row["score"] for row in rows]), "category": category, "stage": stage, "color": color}

def find_kpi(kpis: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for kpi in kpis:
        if kpi["label"] == label:
            return kpi
    for kpi in DEFAULT_KPIS:
        if kpi["label"] == label:
            return kpi
    return DEFAULT_KPIS[0]

def count_by_label(rows: List[Dict[str, Any]], getter) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = getter(row) or "Unknown"
        counts[key] = counts.get(key, 0) + 1
    return [{"label": label, "count": count} for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]

def count_multi_select(rows: List[Dict[str, Any]], column: str) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for row in rows:
        raw = safe(row.get(column))
        if not raw:
            continue
        for item in split_multi_select(raw):
            counts[item] = counts.get(item, 0) + 1
    return [{"label": label, "count": count} for label, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]

def outcome_stats(mapped_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[int]] = {}
    for row in mapped_rows:
        key = (row["category"], row["stage"], row["outcome"])
        groups.setdefault(key, []).append(int(row["score"]))
    items = []
    for (category, stage, outcome), values in groups.items():
        items.append({"category": category, "stage": stage, "outcome": outcome, "value": avg(values), "label": f"{outcome[:52]}{'…' if len(outcome) > 52 else ''}"})
    return sorted(items, key=lambda x: x["value"], reverse=True)

def compute_analytics(filtered_rows: List[Dict[str, Any]], source_row_count: int) -> Dict[str, Any]:
    def by_category_stage(category: str, stage: str) -> List[Dict[str, Any]]:
        return [row for row in filtered_rows if row["category"] == category and row["stage"] == stage]
    kpis = [
        build_kpi("Social Spark", by_category_stage("Social", "Spark"), "Social", "Spark", PALETTE["primary"]),
        build_kpi("Social Growth", by_category_stage("Social", "Growth"), "Social", "Growth", PALETTE["secondary"]),
        build_kpi("Social Horizon", by_category_stage("Social", "Horizon"), "Social", "Horizon", PALETTE["gold"]),
        build_kpi("Cultural Spark", by_category_stage("Cultural", "Spark"), "Cultural", "Spark", PALETTE["accent"]),
        build_kpi("Cultural Growth", by_category_stage("Cultural", "Growth"), "Cultural", "Growth", PALETTE["secondary"]),
        build_kpi("Cultural Horizon", by_category_stage("Cultural", "Horizon"), "Cultural", "Horizon", PALETTE["primary"]),
    ]
    outcomes = outcome_stats(filtered_rows)
    overall = avg([k["value"] for k in kpis])
    strongest = sorted(kpis, key=lambda x: x["value"], reverse=True)[0] if kpis else DEFAULT_KPIS[0]
    weakest = sorted(kpis, key=lambda x: x["value"])[0] if kpis else DEFAULT_KPIS[0]
    completion = round((len(filtered_rows) / (source_row_count * len(MAPPING_RULES))) * 100) if source_row_count else 0
    return {"rows": filtered_rows, "overall": overall, "strongest": strongest, "weakest": weakest, "kpis": kpis, "outcomeStats": outcomes, "dataQuality": {"mappedRows": len(filtered_rows), "sourceRows": source_row_count, "completion": completion}}

def build_bot_reply(prompt: str, analytics: Dict[str, Any]) -> str:

    try:
        context = {
            "overall_score": analytics["overall"],
            "strongest_area": analytics["strongest"]["label"],
            "strongest_value": analytics["strongest"]["value"],
            "weakest_area": analytics["weakest"]["label"],
            "weakest_value": analytics["weakest"]["value"],
        }

        system_prompt = f"""
You are an expert data analyst for Monkey Baa Theatre.

Rules:
- Use ONLY provided data
- Be short and clear
- Always give a recommendation

DATA:
{context}
"""

        import requests
        import os

        api_key = os.getenv("TOGETHER_API_KEY")

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        result = response.json()

# DEBUG SAFETY (VERY IMPORTANT)
        if "choices" not in result:
           return f"API Error: {result}"
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error: {str(e)}"
    
def generate_ai_summary_together(analytics: Dict[str, Any]) -> str:
    import requests
    import os

    api_key = os.getenv("TOGETHER_API_KEY")

    context = {
        "overall": analytics["overall"],
        "strongest": analytics["strongest"]["label"],
        "strongest_value": analytics["strongest"]["value"],
        "weakest": analytics["weakest"]["label"],
        "weakest_value": analytics["weakest"]["value"],
        "kpis": {k["label"]: k["value"] for k in analytics["kpis"]}
    }

    prompt = f"""
You are an expert arts impact analyst.

Write:
- 1 short executive summary paragraph
- 3 bullet insights
- 1 recommendation

Use ONLY this data:
{context}
"""

    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            "messages": [
                {"role": "system", "content": "You are a professional analyst."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
    )

    result = response.json()

    if "choices" not in result:
        return f"AI Summary Error: {result}"

    return result["choices"][0]["message"]["content"]

def create_report_text(analytics: Dict[str, Any], selected_show: str, selected_location: str, filtered_source_rows) -> str:
    date = infer_date(filtered_source_rows)
    quote = extract_teacher_quote(filtered_source_rows)
    strongest = analytics["strongest"]
    weakest = analytics["weakest"]

    social_spark = find_kpi(analytics["kpis"], "Social Spark")["value"]
    social_growth = find_kpi(analytics["kpis"], "Social Growth")["value"]
    social_horizon = find_kpi(analytics["kpis"], "Social Horizon")["value"]

    cultural_spark = find_kpi(analytics["kpis"], "Cultural Spark")["value"]
    cultural_growth = find_kpi(analytics["kpis"], "Cultural Growth")["value"]
    cultural_horizon = find_kpi(analytics["kpis"], "Cultural Horizon")["value"]

    overall = analytics["overall"]

    gap = social_spark - social_growth

    return f"""
Prototype Impact Report (AI-Assisted)

Monkey Baa Theatre Company  
Program: {selected_show}  
Location: {selected_location}  
Date: {date}
Audience: {analytics['dataQuality']['sourceRows']} participants  

------------------------------------------------------------

IMPACT AT A GLANCE
• Engagement (Spark): {social_spark}%
• Cultural Outcomes: {cultural_spark}% – {cultural_horizon}%
• Social Growth: {social_growth}%
• Overall Impact Score: {overall}%

These results show strong immediate engagement and cultural resonance, with a clear opportunity to strengthen deeper social development outcomes.

------------------------------------------------------------

KEY INSIGHT
Students demonstrated high engagement and cultural connection, while Social Growth outcomes were comparatively lower ({social_growth}%). 
This {gap}% gap between Spark and Growth suggests that while the performance captures attention and imagination, additional support is needed to deepen social development.

------------------------------------------------------------

WHAT THIS MEANS
• The program is highly engaging and emotionally impactful  
• Cultural outcomes are consistently strong  
• Social Growth is the weakest area and requires improvement  
• Spark does not automatically translate into Growth without intentional design  

------------------------------------------------------------

WHAT WE’RE DOING NEXT
• Introduce more interactive and collaborative elements  
• Provide structured post-show learning resources  
• Support confidence, communication, and peer interaction  
• Track Social Growth improvements over time  

------------------------------------------------------------

SUPPORTING DATA
• Social Growth: {social_growth}%  
• Social Horizon: {social_horizon}%  
• Cultural Range: {cultural_spark}% – {cultural_horizon}%  
• Overall Impact: {overall}%  

------------------------------------------------------------

This report is AI-assisted and based on survey data analysis aligned with Monkey Baa’s Theory of Change.
"""
def generate_pdf(report_text: str, filename="impact_report.pdf"):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    elements = []

    # Split report into lines
    for line in report_text.split("\n"):
        if line.strip() == "":
            elements.append(Spacer(1, 10))
        else:
            elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 6))

    doc.build(elements)
    return filename
def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), ["Sheet1"]
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    if not sheets:
        raise ValueError("No sheets found")
    first_name = list(sheets.keys())[0]
    return sheets[first_name], list(sheets.keys())

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [safe(col) for col in normalized.columns]
    for col in normalized.columns:
        if normalized[col].dtype == "object":
            normalized[col] = normalized[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    normalized = normalized.dropna(how="all")
    normalized = normalized.loc[normalized.apply(lambda row: any(safe(v) != "" for v in row.values.tolist()), axis=1)]
    return normalized

def kpi_card(label: str, value: int, category: str, stage: str, color: str):
    st.markdown(f"""
        <div style="
            background:white;
            border-radius:16px;
            padding:16px;
            margin-bottom:16px;
        ">
          <div style="font-size:12px;color:#6b7280;">
            {category} · {stage}
          </div>
          <div style="font-size:16px;font-weight:600;margin-top:4px;">
            {label}
          </div>
          <div style="font-size:28px;font-weight:800;color:{color};margin-top:8px;">
            {value}%
          </div>
        </div>
    """, unsafe_allow_html=True)

def bar_rows(items: List[Dict[str, Any]], value_key: str, color: str, pct_suffix: str = ""):
    if not items:
        st.info("No data available.")
        return
    max_value = max(float(item[value_key]) for item in items)
    max_value = max(max_value, 1.0)
    for item in items:
        value = float(item[value_key])
        pct = (value / max_value) * 100
        st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;gap:12px;font-size:14px;margin-bottom:6px;">
                <div style="flex:1;min-width:0;">{item['label']}</div>
                <div style="font-weight:700;">{int(round(value))}{pct_suffix}</div>
              </div>
              <div style="height:14px;background:#f1f5f9;border-radius:999px;overflow:hidden;">
                <div style="width:{pct:.1f}%;height:14px;background:{color};border-radius:999px;"></div>
              </div>
            </div>
        """, unsafe_allow_html=True)

def render_bar_card(
    title: str,
    items: List[Dict[str, Any]],
    value_key: str,
    color: str,
    pct_suffix: str = "",
    caption: str = "",
    margin_top: int = 0,
):
    if margin_top:
        st.markdown(f'<div style="height:{margin_top}px;"></div>', unsafe_allow_html=True)

    key = "white_card_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    with st.container(border=True, key=key):
        st.subheader(title)
        bar_rows(items, value_key, color, pct_suffix)
        if caption:
            st.caption(caption)

def render_plotly_bar_card(
    title: str,
    items: List[Dict[str, Any]],
    color: str,
    caption: str = "",
):
    key = "white_card_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    with st.container(border=True, key=key):
        st.subheader(title)
        if px and items:
            chart_df = pd.DataFrame(items).sort_values(by="count", ascending=True)
            fig = px.bar(
                chart_df,
                x="count",
                y="label",
                orientation="h",
                color_discrete_sequence=[color],
                text="count",
            )
            fig.update_layout(
                height=max(240, 42 * len(chart_df)),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                bargap=0.35,
            )
            fig.update_traces(
                textposition="outside",
                marker=dict(line=dict(width=0)),
                cliponaxis=False,
            )
            fig.update_xaxes(showgrid=False, visible=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            bar_rows(items, "count", color)
        if caption:
            st.caption(caption)

st.set_page_config(page_title="Monkey Baa Impact Dashboard", layout="wide")
st.markdown("""
<style>
.stApp { background: #FFF7E8; }

/* GLOBAL PADDING */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* BANNERS */
.banner-core { 
    background:#EEF6F1;
    border:1px solid #d8efe2;
    border-radius:20px;
    padding:16px 18px;
    margin-bottom:20px;
    line-height:1.5;
}

.banner-support { 
    background:#EEF6FB;
    border:1px solid #d7ebf7;
    border-radius:20px;
    padding:16px 18px;
    margin-bottom:20px;
    line-height:1.5;
}

/* CARDS */
.side-card { 
    background:white;
    border:1px solid #f1f1f1;
    border-radius:16px;
    padding:12px;
    box-shadow:0 1px 6px rgba(0,0,0,0.05);
    margin-bottom:12px;
}

/* KPI SPACING */
.kpi-card {
    margin-bottom:4px;
}

.section-card {
    background: white;
    border-radius: 16px;
    padding: 12px;
    margin-top: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

.main-content-narrow {
    max-width: 980px;
}

div[data-testid="stVerticalBlockBorderWrapper"],
.stApp div[data-testid="stVerticalBlockBorderWrapper"],
.stApp div[data-testid="stVerticalBlockBorderWrapper"] > div,
.stApp div[data-testid="stVerticalBlockBorderWrapper"] > div > div,
.stApp div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-radius: 16px !important;
}

.stApp div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #f1f1f1 !important;
    padding: 12px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
    margin-top: 8px !important;
    margin-bottom: 12px !important;
}

.stApp div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stElementContainer"],
.stApp div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] {
    background: transparent !important;
    background-color: transparent !important;
}

.stApp div[class*="st-key-white_card"],
.stApp div[class*="st-key-white_card"] > div,
.stApp div[class*="st-key-white_card"] > div > div,
.stApp div[class*="st-key-white_card"] div[data-testid="stVerticalBlock"],
.stApp div[class*="st-key-white_card"] div[data-testid="stElementContainer"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-radius: 16px !important;
}

.stApp div[class*="st-key-white_card"] {
    border: 1px solid #f1f1f1 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}

.stApp div[class*="st-key-white_card_insight_bot"],
.stApp div[class*="st-key-white_card_insight_bot"] > div,
.stApp div[class*="st-key-white_card_insight_bot"] > div > div,
.stApp div[class*="st-key-white_card_insight_bot"] div[data-testid="stVerticalBlock"],
.stApp div[class*="st-key-white_card_insight_bot"] div[data-testid="stElementContainer"],
.stApp div[class*="st-key-white_card_quality_summary"],
.stApp div[class*="st-key-white_card_quality_summary"] > div,
.stApp div[class*="st-key-white_card_quality_summary"] > div > div,
.stApp div[class*="st-key-white_card_quality_summary"] div[data-testid="stVerticalBlock"],
.stApp div[class*="st-key-white_card_quality_summary"] div[data-testid="stElementContainer"] {
    background: #FFFBF1 !important;
    background-color: #FFFBF1 !important;
}

/* COLUMN GAP FIX */
div[data-testid="column"] {
    padding-left:4px;
    padding-right:4px;
}

/* CHAT */
.chat-bubble {
    margin-bottom:10px;
}

.overall-score-card {
    background:#EAF7F4;
    border:1px solid #ccebe4;
    border-radius:14px;
    padding:10px 14px;
    text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,0.05);
    max-width:160px;
    margin-left:auto;
}

.overall-score-label {
    font-size:11px;
    font-weight:700;
    color:#0f766e;
    text-transform:uppercase;
    letter-spacing:0;
}

.overall-score-value {
    font-size:28px;
    font-weight:900;
    color:#14B8A6;
    line-height:1.05;
    margin-top:3px;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    div[data-testid="column"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    .main-content-narrow {
        max-width: 100%;
    }

    .banner-core,
    .banner-support {
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .stApp div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 10px !important;
        margin-top: 8px !important;
        margin-bottom: 10px !important;
        border-radius: 14px !important;
    }

    .stApp div[class*="st-key-white_card"],
    .stApp div[class*="st-key-white_card"] > div,
    .stApp div[class*="st-key-white_card"] > div > div,
    .stApp div[class*="st-key-white_card"] div[data-testid="stVerticalBlock"] {
        border-radius: 14px !important;
    }

    h1 {
        font-size: 1.65rem !important;
    }

    h2, h3 {
        font-size: 1.05rem !important;
    }

    div[data-testid="stMetric"] {
        width: 100%;
    }

    .overall-score-card {
        max-width: 100%;
        margin-left: 0;
        margin-top: 10px;
        padding: 12px;
    }

    .overall-score-value {
        font-size: 26px;
    }
}

</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": BOT_INTRO}]

st.title("Monkey Baa Impact Dashboard")
st.caption("Framework-aligned analysis of social and cultural outcomes")
uploaded_file = st.file_uploader("Upload a CSV or Excel survey export", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("Upload a CSV or Excel survey export and analyze it using the Monkey Baa outcome framework.")
    st.stop()

with st.spinner("Processing your file..."):
    df, sheets = load_uploaded_file(uploaded_file)
    df = normalize_dataframe(df)
    if df.empty:
        st.error("No rows found in the uploaded file.")
        st.stop()
    missing_columns = [col for col in get_required_columns() if col not in df.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        st.stop()
    source_rows = df.to_dict(orient="records")
    all_mapped_rows = map_survey_rows(source_rows)

if not all_mapped_rows:
    st.error("The file loaded, but no rows could be mapped into the framework.")
    st.stop()

unique_shows = sorted({row["show"] for row in all_mapped_rows})

if "Where is the Green Sheep" not in unique_shows:
    unique_shows.append("Where is the Green Sheep")

shows = ["All shows"] + unique_shows

default_show = shows[0]
base_rows = all_mapped_rows

c1, c2, c3 = st.columns(3)
with c1:
    selected_show = st.selectbox("Show", shows, index=0, key="show_filter_main")
if selected_show == "All shows":
    base_rows = all_mapped_rows
else:
    base_rows = [row for row in all_mapped_rows if row["show"] == selected_show]
audiences = ["All"] + sorted({row["audience"] for row in base_rows})
locations = ["All"] + sorted({row["location"] for row in base_rows})
with c2:
    filter_audience = st.selectbox("Audience", audiences, key="audience_filter_main")
with c3:
    filter_location = st.selectbox("Location", locations, key="location_filter_main")

filtered_rows = [row for row in base_rows if (filter_audience == "All" or row["audience"] == filter_audience) and (filter_location == "All" or row["location"] == filter_location)]
filtered_source_rows = [
    row for row in source_rows
    if (
        (selected_show == "All shows" or normalize_show_name(infer_show(row)) == selected_show)
        and (filter_audience == "All" or infer_audience(row) == filter_audience)
        and (filter_location == "All" or infer_location(row) == filter_location)
    )
]

analytics = compute_analytics(filtered_rows, len(source_rows))
with st.spinner("Generating insights..."):
    
   report_text = create_report_text(
    analytics,
    selected_show,
    filter_location,
    filtered_source_rows
)

top_left, top_mid, top_right = st.columns([1, 2, 1])
with top_left:
    col1, col2, col3 = st.columns(3)

    # TXT DOWNLOAD
    with col1:
        st.download_button(
            "Download TXT",
            data=report_text.encode("utf-8"),
            file_name="impact_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    # PDF DOWNLOAD 🔥
    with col2:
        if st.button("Generate PDF", use_container_width=True):
            pdf_file = generate_pdf(report_text)

            with open(pdf_file, "rb") as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name="impact_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    # PRINT
    with col3:
        if st.button("Print Report", use_container_width=True):
            html_content = f"""
            <html>
            <body style="font-family:Arial;padding:40px;white-space:pre-wrap;">
            {report_text}
            <script>
                window.onload = function() {{
                    window.print();
                }}
            </script>
            </body>
            </html>
            """
            st.components.v1.html(html_content, height=0)
with top_mid:
    st.markdown(f"""
    <div style="text-align:center;">
      <div style="font-size:14px;font-weight:600;color:#6b7280;">Monkey Baa Outcome Dashboard</div>
      <div style="font-size:32px;font-weight:900;">Framework-Aligned Impact Analysis</div>
      <div style="margin-top:6px;font-size:14px;color:#6b7280;">{uploaded_file.name} · {len(source_rows)} rows · {len(df.columns)} columns · {len(sheets)} sheet(s){'' if selected_show == 'All shows' else ' · Filter: ' + selected_show}</div>
    </div>
    """, unsafe_allow_html=True)
with top_right:
    st.markdown(f"""
    <div class="overall-score-card">
      <div class="overall-score-label">Overall</div>
      <div class="overall-score-value">{analytics['overall']}%</div>
    </div>
    """, unsafe_allow_html=True)

main_col, side_col = st.columns([3.2, 1.25], gap="large")

with main_col:
    st.markdown('<div class="main-content-narrow">', unsafe_allow_html=True)
    st.markdown("""
    <div class="banner-core">
        <div style="font-size:13px;font-weight:700;text-transform:uppercase;">
            Core Impact Charts
        </div>
        <div style="margin-top:4px;font-size:14px;color:#4b5563;">
            These are framework-critical metrics directly mapped to Monkey Baa’s Theory of Change (Spark → Growth → Horizon). Use these for decision-making and reporting.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS
    kpi_cols = st.columns(3)
    for idx, kpi in enumerate(analytics["kpis"]):
        with kpi_cols[idx % 3]:
            kpi_card(kpi["label"], kpi["value"], kpi["category"], kpi["stage"], kpi["color"])

    # 🔥 KEY INSIGHT (SIMPLE TEXT — NO CARD)
    strongest = analytics["strongest"]

    st.markdown(f"""
    <div style="margin-top:10px;font-size:14px;color:#374151;">
    <strong>Key Insight:</strong> The strongest outcome area is <b>{strongest['label']}</b> at <b>{strongest['value']}%</b>, indicating this is where the program delivers the most impact.
    </div>
    """, unsafe_allow_html=True)

    # 🔥 OUTCOME COMPARISON (CLEAN CARD)
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    spark = find_kpi(analytics["kpis"], "Social Spark")["value"]
    growth = find_kpi(analytics["kpis"], "Social Growth")["value"]
    horizon = find_kpi(analytics["kpis"], "Social Horizon")["value"]
    gap = horizon - spark
    comparison_data = [
        {"label": "Spark", "value": spark},
        {"label": "Growth", "value": growth},
        {"label": "Horizon", "value": horizon},
    ]

    with st.container(border=True, key="white_card_outcome_comparison"):
       st.markdown("### Outcome Comparison: Spark vs Growth vs Horizon")
       st.markdown(f"""
       <div style="font-size:14px;color:#4b5563;margin-top:6px;">
       There is a <b>{gap}% gap</b> between initial engagement (Spark) and long-term impact (Horizon), indicating strong sustained outcomes beyond first impressions.
       </div>
       """, unsafe_allow_html=True)
       bar_rows(comparison_data, "value", PALETTE["primary"], "%")
       st.caption("This chart highlights the performance gap between Spark, Growth, and Horizon outcomes.")
    col_left, col_right = st.columns(2)

# 🔵 SOCIAL (LEFT)
    with col_left:
       for stage in ["Spark", "Growth", "Horizon"]:
           subset = [
               item for item in analytics["outcomeStats"]
               if item["category"] == "Social" and item["stage"] == stage
           ]

           with st.container(border=True, key=f"white_card_social_{stage.lower()}"):
               st.subheader(f"Social {stage} Outcomes")
               if subset:
                   bar_rows(subset, "value", STAGE_COLORS[("Social", stage)], "%")
               else:
                   st.info("No data available.")


# 🟣 CULTURAL (RIGHT)
    with col_right:
       for stage in ["Spark", "Growth", "Horizon"]:
           subset = [
               item for item in analytics["outcomeStats"]
               if item["category"] == "Cultural" and item["stage"] == stage
           ]

           with st.container(border=True, key=f"white_card_cultural_{stage.lower()}"):
               st.subheader(f"Cultural {stage} Outcomes")
               if subset:
                   bar_rows(subset, "value", STAGE_COLORS[("Cultural", stage)], "%")
               else:
                   st.info("No data available.")

    st.markdown('<div class="banner-support" style="margin-top:16px;"><div style="font-size:13px;font-weight:700;text-transform:uppercase;">Supporting Impact Charts</div><div style="margin-top:4px;font-size:14px;color:#4b5563;">Contextual metrics that support interpretation of core outcomes, including response distribution and engagement patterns.</div></div>', unsafe_allow_html=True)

    show_counts = count_by_label(filtered_source_rows, infer_show)
    behaviour_counts = count_multi_select(filtered_source_rows, "After the show, did the young person... multiplechoice")
    emotion_counts = count_multi_select(filtered_source_rows, "What feeling/s did the young person experience during the performance? multiplechoice")
    audience_counts = count_by_label(filtered_source_rows, infer_audience)
    location_counts = count_by_label(filtered_source_rows, infer_location)

    s1, s2 = st.columns(2)

    with s1:
       render_bar_card(
           "Distribution by Show",
           show_counts,
           "count",
           PALETTE["primary"]
       )

    with s2:
       render_plotly_bar_card(
           "Audience Segmentation",
            audience_counts,
            PALETTE["secondary"]
       )


    s3, s4 = st.columns(2)
    with s3:
        render_plotly_bar_card(
           "Behaviour Distribution",
           behaviour_counts,
           PALETTE["accent"],
           caption="Actions participants took after the performance."
       )
        
    with s4:
        render_plotly_bar_card(
            "Emotion Distribution",
            emotion_counts,
            PALETTE["primary"],
            caption="This chart shows the distribution of emotions experienced by participants.",
        )
        


# 🔥 SENTIMENT ANALYSIS (NOW OUTSIDE s4)
    sentiment_counts = {
       "Positive Experience": 0,
       "Mixed Response": 0,
       "Could Be Improved": 0
    }

    for item in emotion_counts:
       label = item["label"].lower()

       if any(word in label for word in ["happy", "excited", "inspired", "curious", "joy", "amazing", "fun"]):
           sentiment_counts["Positive Experience"] += item["count"]

       elif any(word in label for word in ["confused", "bored", "unsure", "neutral"]):
           sentiment_counts["Mixed Response"] += item["count"]

       else:
           sentiment_counts["Could Be Improved"] += item["count"]
    
    sentiment_data = [{"label": k, "count": v} for k, v in sentiment_counts.items()]
    render_bar_card(
        "Sentiment Analysis of Participant Feedback",
        sentiment_data,
        "count",
        PALETTE["secondary"],
        caption="This chart groups participant responses into overall experience categories.",
    )



    render_bar_card("Location Distribution", location_counts, "count", PALETTE["accent"], margin_top=16)

    if False and px and location_counts:
       df_loc = pd.DataFrame(location_counts).sort_values(by="count", ascending=True)

       fig = px.bar(
           df_loc,
           x="count",
           y="label",
           orientation="h",
           color_discrete_sequence=[PALETTE["accent"]]
       )

       fig.update_layout(
           height=320,
           margin=dict(l=10, r=10, t=10, b=10),
           xaxis_title="",
           yaxis_title="",
           plot_bgcolor="white",
           paper_bgcolor="white",
           font=dict(size=13),
           showlegend=False
       )

       fig.update_traces(
           text=df_loc["count"],
           textposition="outside",
           marker=dict(
              line=dict(width=0)
           )
       )

       fig.update_xaxes(showgrid=False)
       fig.update_yaxes(showgrid=False)

       st.plotly_chart(fig, use_container_width=True)

    else:
       pass


    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="white_card_mapped_survey_data"):
        st.subheader("Mapped Survey Data")
        tf1, tf2, tf3 = st.columns(3)
        table_audience_options = ["All"] + sorted({row["audience"] for row in filtered_rows})
        with tf1:
            table_audience = st.selectbox("Audience filter", table_audience_options, key="table_audience")
        with tf2:
            table_category = st.selectbox("Category filter", ["All", "Social", "Cultural"], key="table_category")
        with tf3:
            table_stage = st.selectbox("Stage filter", ["All", "Spark", "Growth", "Horizon"], key="table_stage")

        filtered_table_rows = [row for row in filtered_rows if (table_audience == "All" or row["audience"] == table_audience) and (table_category == "All" or row["category"] == table_category) and (table_stage == "All" or row["stage"] == table_stage)]
        page_size = 10
        page_count = max(1, math.ceil(len(filtered_table_rows) / page_size))
        page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
        start = (page - 1) * page_size
        table_df = pd.DataFrame(filtered_table_rows[start:start + page_size])
        if not table_df.empty:
            st.dataframe(table_df[["audience", "show", "location", "outcome", "category", "stage", "score"]], use_container_width=True, hide_index=True)
        else:
            st.info("No mapped data available.")
        st.caption(f"Page {page} of {page_count}")
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    with st.container(border=True, key="white_card_insight_bot"):
        st.subheader("Gen AI Insight Bot")
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                st.markdown(f'<div style="white-space:pre-line;border:1px solid #f6d3c7;background:white;border-radius:18px;padding:12px;margin-bottom:10px;">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="white-space:pre-line;background:#4DB7E5;color:white;border-radius:18px;padding:12px;margin-bottom:10px;margin-left:24px;">{message["content"]}</div>', unsafe_allow_html=True)
        bot_input = st.text_area("Ask the bot for insights", height=120, placeholder="e.g. What is the weakest outcome and how can it be improved?")
        if st.button("Get insight", use_container_width=True) and bot_input.strip():
            st.session_state.messages.append({"role": "user", "content": bot_input.strip()})
            st.session_state.messages.append({"role": "assistant", "content": build_bot_reply(bot_input.strip(), analytics)})
            st.rerun()

        st.caption("Suggested questions:")
        for question in PROMPT_SUGGESTIONS:
            if st.button(question, key=question, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": build_bot_reply(question, analytics)})
                st.rerun()

    with st.container(border=True, key="white_card_quality_summary"):
        st.markdown("**Framework Quality Summary**")
        st.write(f"**Source rows:** {analytics['dataQuality']['sourceRows']}")
        st.write(f"**Mapped records:** {analytics['dataQuality']['mappedRows']}")
        st.write(f"**Approx. mapping completion:** {analytics['dataQuality']['completion']}%")
        st.write(f"**Show filter:** {selected_show}")
        st.write(f"**Location filter:** {filter_location}")
        st.write(f"**Strongest area:** {analytics['strongest']['label']}")
        st.write(f"**Weakest area:** {analytics['weakest']['label']}")