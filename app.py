#!/usr/bin/env python3
"""
MEKANSAL VERI — Birlesik Demo Arayuzu

Tum demolar tek arayuzde. Coklu format destegi, lokasyon girisi, interaktif harita.

Calistirma:
  streamlit run app.py
  streamlit run app.py --server.port 8501
"""

import sys
import os
import time
import tempfile
import shutil
import json
import re
from pathlib import Path
from collections import Counter
from html import escape
from urllib.parse import urlencode
from urllib.request import urlopen

import streamlit as st
import numpy as np

APP_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(APP_DIR))

from utils import (
    SUPPORTED_EXTENSIONS,
    RASTER_EXTENSIONS,
    VECTOR_EXTENSIONS,
    LOCATIONS,
    SAMPLE_DATA,
    to_raster,
    to_vector,
    is_raster,
    get_image_info,
    geocode_location,
    download_location_imagery,
)


def _norm_rgb(arr):
    arr = arr.astype(np.float32)
    for i in range(arr.shape[2]):
        band = arr[:, :, i]
        valid = band[band > 0]
        if len(valid) > 0:
            p2, p98 = np.nanpercentile(valid, [2, 98])
            if p98 - p2 > 0:
                arr[:, :, i] = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
            else:
                arr[:, :, i] = 0
    return arr.astype(np.uint8)


def _arcgis_get_json(url: str, params: dict[str, object]) -> dict:
    query_string = urlencode(params)
    with urlopen(f"{url}?{query_string}") as response:
        return json.load(response)


@st.cache_data(show_spinner=False, ttl=1800)
def _find_public_feature_service(search_query: str) -> dict:
    payload = _arcgis_get_json(
        "https://www.arcgis.com/sharing/rest/search",
        {
            "q": search_query,
            "sortField": "numviews",
            "sortOrder": "desc",
            "num": 10,
            "f": "json",
        },
    )
    for result in payload.get("results", []):
        if result.get("type") == "Feature Service" and result.get("url"):
            return result
    raise RuntimeError(f"Public feature service bulunamadı: {search_query}")


@st.cache_data(show_spinner=False, ttl=1800)
def _find_point_layer_url(service_url: str) -> tuple[str, str]:
    payload = _arcgis_get_json(service_url, {"f": "json"})
    for layer in payload.get("layers", []):
        layer_id = layer.get("id")
        if layer_id is None:
            continue
        layer_url = f"{service_url}/{layer_id}"
        layer_meta = _arcgis_get_json(layer_url, {"f": "json"})
        if layer_meta.get("geometryType") == "esriGeometryPoint":
            return layer_url, layer_meta.get("name", f"Layer {layer_id}")
    raise RuntimeError("Servis içinde point geometry layer bulunamadı.")


@st.cache_data(show_spinner=False, ttl=1800)
def _query_point_features(layer_url: str, max_records: int) -> list[tuple[float, float, dict]]:
    payload = _arcgis_get_json(
        f"{layer_url}/query",
        {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "json",
            "resultRecordCount": max_records,
            "outSR": 4326,
        },
    )
    points = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        x = geometry.get("x")
        y = geometry.get("y")
        if x is None or y is None:
            continue
        points.append((y, x, feature.get("attributes") or {}))
    if not points:
        raise RuntimeError("Sorgudan nokta geometri dönmedi.")
    return points


LLM_API_URL = "http://192.168.1.200:8001/v1/chat/completions"
LLM_MODEL = "moonshotai/Kimi-K2.6"
# LLM_API_URL = "http://192.168.1.200:8000/v1/chat/completions"
# LLM_MODEL = "GLM-5.1-FP8"

FIELD_HINTS = {
    "crime_type": [
        "primary_type", "crime_type", "offense", "offense_type", "category",
        "incident_type", "ucr_desc", "violation", "type",
    ],
    "description": [
        "description", "desc", "details", "summary", "subtype",
        "secondary", "location_description", "crime_desc",
    ],
    "city": ["city", "municipality", "town", "village"],
    "district": [
        "district", "beat", "ward", "community_area", "community",
        "neighborhood", "region", "county", "state",
    ],
    "date": ["date", "incident_date", "reported_date", "report_date", "datetime", "timestamp"],
}

VEHICLE_THEFT_TERMS = [
    "vehicle theft", "motor vehicle theft", "stolen vehicle", "stolen auto",
    "auto theft", "car theft", "araba hirsizligi", "arac hirsizligi",
    "otomobil hirsizligi", "vehicle", "auto", "car",
]


def _normalize_text(value) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = text.replace("ı", "i").replace("İ", "i").replace("ş", "s").replace("ğ", "g")
    text = text.replace("ü", "u").replace("ö", "o").replace("ç", "c")
    text = re.sub(r"\s+", " ", text)
    return text


def _safe_label(value) -> str:
    if value is None:
        return "Bilinmiyor"
    text = str(value).strip()
    return text if text else "Bilinmiyor"


def _find_best_field(field_names: list[str], hint_keys: list[str]) -> str | None:
    scored = []
    for field in field_names:
        norm = _normalize_text(field)
        score = 0
        for idx, hint in enumerate(hint_keys):
            if norm == hint:
                score += 100 - idx
            elif hint in norm:
                score += 50 - idx
        if score > 0:
            scored.append((score, field))
    return max(scored)[1] if scored else None


def _infer_category_field(points: list[tuple[float, float, dict]]) -> str | None:
    _, _, sample = points[0]
    skip = {"objectid", "fid", "oid", "id", "globalid", "shape", "geometry",
            "xcoord", "ycoord", "latitude", "longitude", "the_geom"}
    candidates = []
    for field in sorted(sample.keys()):
        norm = _normalize_text(field)
        if any(s in norm for s in skip):
            continue
        values = [attrs.get(field) for _, _, attrs in points]
        non_null = [v for v in values if v is not None and str(v).strip()]
        if len(non_null) < max(3, len(points) * 0.3):
            continue
        str_values = [str(v).strip() for v in non_null]
        unique = len(set(str_values))
        ratio = unique / len(str_values) if str_values else 0
        if all(_is_number(v) for v in str_values[:30]):
            continue
        if ratio < 0.01 or ratio > 0.95:
            continue
        avg_len = sum(len(v) for v in str_values) / len(str_values)
        if avg_len > 80:
            continue
        candidates.append((field, ratio, unique, avg_len))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[2] if c[2] < 30 else 30, c[3]))
    return candidates[0][0]


def _infer_location_field(points: list[tuple[float, float, dict]], hint_keys: list[str]) -> str | None:
    name_match = _find_best_field(
        sorted({k for _, _, a in points for k in a.keys()}), hint_keys
    )
    if name_match:
        values = [str(attrs.get(name_match, "")).strip() for _, _, attrs in points]
        non_null = [v for v in values if v and v != "None"]
        unique = len(set(non_null))
        ratio = unique / len(non_null) if non_null else 0
        if 0.01 < ratio < 0.95 and unique < len(points) * 0.8:
            return name_match
    for field in sorted({k for _, _, a in points for k in a.keys()}):
        if _find_best_field([field], hint_keys):
            continue
        values = [str(attrs.get(field, "")).strip() for _, _, attrs in points]
        non_null = [v for v in values if v and v != "None"]
        if len(non_null) < max(3, len(points) * 0.3):
            continue
        unique = len(set(non_null))
        ratio = unique / len(non_null) if non_null else 0
        if all(_is_number(v) for v in non_null[:30]):
            continue
        if 0.01 < ratio < 0.3 and unique < len(points) * 0.5:
            return field
    return None


def _infer_date_field(points: list[tuple[float, float, dict]]) -> str | None:
    date_patterns = [
        re.compile(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"),
        re.compile(r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}"),
    ]
    for field in sorted({k for _, _, a in points for k in a.keys()}):
        norm = _normalize_text(field)
        if any(t in norm for t in ["objectid", "id", "globalid", "shape"]):
            continue
        values = [attrs.get(field) for _, _, attrs in points]
        non_null = [v for v in values if v is not None and str(v).strip()]
        if len(non_null) < max(3, len(points) * 0.3):
            continue
        str_values = [str(v).strip() for v in non_null[:20]]
        date_count = sum(1 for v in str_values if any(p.search(v) for p in date_patterns))
        if date_count > len(str_values) * 0.5:
            return field
        if isinstance(non_null[0], (int, float)) and non_null[0] > 1e12:
            return field
    return _find_best_field(
        sorted({k for _, _, a in points for k in a.keys()}), FIELD_HINTS["date"]
    )


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _top_field_values(points: list[tuple[float, float, dict]], field_name: str | None, limit: int = 12) -> list[str]:
    if not field_name:
        return []
    counter = Counter()
    for _, _, attrs in points:
        value = attrs.get(field_name)
        if value not in (None, ""):
            counter[_safe_label(value)] += 1
    return [value for value, _ in counter.most_common(limit)]


def _build_dataset_profile(points: list[tuple[float, float, dict]]) -> dict:
    field_names = sorted({key for _, _, attrs in points for key in attrs.keys()})
    category_field = _infer_category_field(points)
    if category_field:
        hints_match = _find_best_field(field_names, FIELD_HINTS["crime_type"])
        crime_type = hints_match if hints_match else category_field
    else:
        crime_type = _find_best_field(field_names, FIELD_HINTS["crime_type"])
    inferred = {
        "crime_type": crime_type,
        "description": _find_best_field(field_names, FIELD_HINTS["description"]),
        "city": _infer_location_field(points, FIELD_HINTS["city"]),
        "district": _infer_location_field(points, FIELD_HINTS["district"]),
        "date": _infer_date_field(points),
    }
    if inferred["city"] == inferred["district"]:
        inferred["district"] = None
    profile = {
        "field_names": field_names,
        "inferred": inferred,
        "top_crime_values": _top_field_values(points, inferred["crime_type"]),
        "top_city_values": _top_field_values(points, inferred["city"]),
        "top_district_values": _top_field_values(points, inferred["district"]),
        "record_count": len(points),
        "sample_rows": [],
    }
    for _, _, attrs in points[:3]:
        sample = {}
        for key in field_names[:8]:
            if key in attrs:
                sample[key] = attrs[key]
        profile["sample_rows"].append(sample)
    return profile


def _build_field_overview(points: list[tuple[float, float, dict]]) -> list[dict]:
    if not points:
        return []
    skip_norms = {
        "objectid", "shape", "geometry", "fid", "oid",
        "x_coordinate", "y_coordinate", "latitude", "longitude",
        "xcoord", "ycoord", "gdb_geomattr_data", "globalid",
        "creationdate", "creator", "editdate", "editor",
    }
    all_fields = sorted({key for _, _, attrs in points for key in attrs.keys()})
    overview = []
    for field in all_fields:
        norm = _normalize_text(field)
        if norm in skip_norms:
            continue
        values = []
        for _, _, attrs in points:
            v = attrs.get(field)
            if v is not None and str(v).strip() and str(v).strip() != "None":
                values.append(v)
        if not values:
            continue
        counter = Counter(_safe_label(v) for v in values)
        is_numeric = all(isinstance(v, (int, float)) for v in values[:30]) if values else False
        overview.append({
            "field": field,
            "type": "sayi" if is_numeric else "metin",
            "non_null": len(values),
            "unique": len(counter),
            "top_values": [v for v, _ in counter.most_common(5)],
        })
    return overview


def _call_llm(messages: list[dict], temperature: float = 0.1, max_tokens: int = 2048) -> str:
    import requests as req

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = req.post(LLM_API_URL, json=payload, timeout=120)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if content is None:
        raise ValueError("LLM content None dondurdu")
    return content


def _extract_json_block(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_question_plan(question: str, profile: dict) -> dict:
    q = _normalize_text(question)
    plan = {
        "intent": "summary",
        "crime_keywords": [],
        "location_text": "",
        "group_by": "crime_type",
        "limit": 10,
    }
    if any(token in q for token in ["araba", "arac", "oto", "vehicle", "auto", "car"]):
        plan["crime_keywords"] = VEHICLE_THEFT_TERMS
        plan["intent"] = "top_locations"
        plan["group_by"] = "city" if profile["inferred"].get("city") else "district"
    if any(token in q for token in ["hangi suclar", "hangi suçlar", "what crimes", "ne sucu", "ne suç"]):
        plan["intent"] = "top_crimes"
        plan["group_by"] = "crime_type"
    if any(token in q for token in ["hangi sehir", "hangi şehir", "where", "nerede", "hangi bolge", "hangi bölge"]):
        plan["intent"] = "top_locations"
        plan["group_by"] = "city" if profile["inferred"].get("city") else "district"
    for value in profile.get("top_city_values", []) + profile.get("top_district_values", []):
        if _normalize_text(value) and _normalize_text(value) in q:
            plan["location_text"] = value
            break
    return plan


def _plan_question(question: str, profile: dict) -> dict:
    schema_payload = {
        "inferred_fields": profile["inferred"],
        "top_city_values": profile["top_city_values"][:10],
        "top_district_values": profile["top_district_values"][:10],
        "top_crime_values": profile["top_crime_values"][:10],
    }
    prompt = f"""
You convert the user's question into a compact JSON query plan for a crime map dataset.
Return only valid JSON.

Allowed intent values: summary, top_crimes, top_locations, count
Allowed group_by values: crime_type, city, district, none
Rules:
- If the user asks about vehicle or car theft, set crime_keywords to vehicle theft synonyms.
- If a location is mentioned, copy it into location_text.
- Prefer city, otherwise district.
- Do not invent fields outside the schema.

Schema:
{json.dumps(schema_payload, ensure_ascii=False)}

User question:
{question}

JSON shape:
{{
  "intent": "summary",
  "crime_keywords": [],
  "location_text": "",
  "group_by": "crime_type",
  "limit": 10
}}
""".strip()
    try:
        content = _call_llm(
            [
                {"role": "system", "content": "You are a strict JSON planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        plan = _extract_json_block(content)
        if plan:
            return {
                "intent": plan.get("intent", "summary"),
                "crime_keywords": plan.get("crime_keywords", []) or [],
                "location_text": plan.get("location_text", "") or "",
                "group_by": plan.get("group_by", "crime_type"),
                "limit": int(plan.get("limit", 10) or 10),
            }
    except Exception:
        pass
    return _fallback_question_plan(question, profile)


def _row_matches_text(attrs: dict, fields: list[str], terms: list[str]) -> bool:
    if not terms:
        return True
    haystack = " ".join(_normalize_text(attrs.get(field, "")) for field in fields if field)
    return any(_normalize_text(term) in haystack for term in terms if term)


def _location_matches(attrs: dict, profile: dict, location_text: str) -> bool:
    if not location_text:
        return True
    fields = [profile["inferred"].get("city"), profile["inferred"].get("district")]
    haystack = " ".join(_normalize_text(attrs.get(field, "")) for field in fields if field)
    return _normalize_text(location_text) in haystack


def _execute_question_plan(points: list[tuple[float, float, dict]], profile: dict, plan: dict) -> dict:
    crime_fields = [profile["inferred"].get("crime_type"), profile["inferred"].get("description")]
    matched = []
    for lat, lon, attrs in points:
        if not _location_matches(attrs, profile, plan.get("location_text", "")):
            continue
        if not _row_matches_text(attrs, crime_fields, plan.get("crime_keywords", [])):
            continue
        matched.append((lat, lon, attrs))

    group_key = plan.get("group_by", "crime_type")
    resolved_group_field = None
    if group_key == "crime_type":
        resolved_group_field = profile["inferred"].get("crime_type")
    elif group_key == "city":
        resolved_group_field = profile["inferred"].get("city")
    elif group_key == "district":
        resolved_group_field = profile["inferred"].get("district")

    grouped = []
    if resolved_group_field:
        counter = Counter()
        for _, _, attrs in matched:
            counter[_safe_label(attrs.get(resolved_group_field))] += 1
        grouped = [{"name": name, "count": count} for name, count in counter.most_common(plan.get("limit", 10))]

    top_crimes = []
    crime_field = profile["inferred"].get("crime_type")
    if crime_field:
        crime_counter = Counter()
        for _, _, attrs in matched:
            crime_counter[_safe_label(attrs.get(crime_field))] += 1
        top_crimes = [{"name": name, "count": count} for name, count in crime_counter.most_common(10)]

    top_locations = []
    loc_field = profile["inferred"].get("city") or profile["inferred"].get("district")
    if loc_field:
        loc_counter = Counter()
        for _, _, attrs in matched:
            loc_counter[_safe_label(attrs.get(loc_field))] += 1
        top_locations = [{"name": name, "count": count} for name, count in loc_counter.most_common(10)]

    sample_records = []
    preview_fields = [
        profile["inferred"].get("crime_type"),
        profile["inferred"].get("description"),
        profile["inferred"].get("city"),
        profile["inferred"].get("district"),
        profile["inferred"].get("date"),
    ]
    for _, _, attrs in matched[:5]:
        row = {}
        for field in preview_fields:
            if field:
                row[field] = attrs.get(field)
        sample_records.append(row)

    return {
        "matched_count": len(matched),
        "total_count": len(points),
        "grouped": grouped,
        "top_crimes": top_crimes,
        "top_locations": top_locations,
        "sample_records": sample_records,
        "matched_points": matched[:200],
    }


def _fallback_answer(question: str, plan: dict, result: dict) -> str:
    lines = [f"Toplam {result['matched_count']} kayıt eşleşti."]
    if plan.get("location_text"):
        lines.append(f"Konum filtresi: {plan['location_text']}.")
    if plan.get("crime_keywords"):
        lines.append("Suç filtresi uygulandı.")
    if result["grouped"]:
        top = ", ".join(f"{item['name']} ({item['count']})" for item in result["grouped"][:5])
        lines.append(f"Öne çıkan dağılım: {top}.")
    elif result["top_crimes"]:
        top = ", ".join(f"{item['name']} ({item['count']})" for item in result["top_crimes"][:5])
        lines.append(f"En sık suçlar: {top}.")
    if result["matched_count"] == 0:
        lines = ["Soruya uyan kayıt bulunamadı. Farklı bir şehir, bölge veya suç ifadesi deneyin."]
    return "\n".join(lines)


def _answer_question(question: str, profile: dict, plan: dict, result: dict) -> str:
    summary_payload = {
        "question": question,
        "plan": plan,
        "matched_count": result["matched_count"],
        "total_count": result["total_count"],
        "top_crimes": result["top_crimes"][:5],
        "top_locations": result["top_locations"][:5],
        "grouped": result["grouped"][:8],
        "sample_records": result["sample_records"][:3],
        "inferred_fields": profile["inferred"],
    }
    prompt = f"""
Use only the supplied analysis result. Answer in Turkish.
Be precise, short, and data-grounded. If the result is empty, say no matching records were found.
Mention which location or crime filter was used when relevant.

Analysis:
{json.dumps(summary_payload, ensure_ascii=False)}
""".strip()
    try:
        return _call_llm(
            [
                {"role": "system", "content": "You answer questions about map data using only supplied analysis."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception:
        return _fallback_answer(question, plan, result)


def _build_folium_map(points: list[tuple[float, float, dict]], title_fields: list[str] | None = None):
    import folium
    from folium.plugins import MarkerCluster, Fullscreen, MeasureControl, MiniMap

    center_lat = sum(lat for lat, _, _ in points) / len(points)
    center_lon = sum(lon for _, lon, _ in points) / len(points)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB positron",
        zoom_snap=0.25,
        zoom_delta=0.5,
        max_zoom=20,
        prefer_canvas=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", max_zoom=20).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Karanlık Tema", max_zoom=20).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    Fullscreen(position="topright", title="Tam ekran", title_cancel="Çık").add_to(m)
    MeasureControl(position="bottomleft", primary_length_unit="kilometers", primary_area_unit="sqmeters").add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    cluster = MarkerCluster().add_to(m)
    for lat, lon, attrs in points:
        popup_lines = []
        fields = title_fields or list(attrs.keys())[:8]
        for key in fields[:8]:
            popup_lines.append(f"<b>{escape(str(key))}</b>: {escape(str(attrs.get(key)))}")
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#c2410c",
            fill=True,
            fill_color="#ea580c",
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup("<br>".join(popup_lines), max_width=400),
            tooltip=escape(str(attrs.get(fields[0], "")))[:80] if fields else None,
        ).add_to(cluster)
    return m


DEMOS = {
    "01 — Metinle Segmentasyon": " ",
    "02 — Görüntü Analizi": "VLM ile Uydu Görüntüsü Analizi",
    "03 — Ajan Analizi": "AI Ajan ile Uçtan Uca Analiz",
    "04 — Hasar Tespiti": "Bina Hasar Sınıflandırma",
    "05 — Süper Çözünürlük": "4x Uydu Görüntüsü Süper Çözünürlük",
    "06 — Değişim Tespiti": "Değişim Tespiti",
    "07 — Folium Harita": "Suç haritasını interaktif haritada göster",
}

st.set_page_config(
    page_title="Mekansal Veri Demo",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛰️ Mekansal Veri")
    st.caption("Yapay Zeka ile Mekansal Analiz Platformu")

    demo_choice = st.selectbox(
        "Demo Seç",
        list(DEMOS.keys()),
        format_func=lambda k: k,
    )
    st.caption(DEMOS[demo_choice])

    st.divider()

    st.subheader("📁 Konum Seç")
    loc_method = st.radio(
        "Nasıl?", ["Şehir Seç", "Koordinat Gir", "Haritadan Seç"], horizontal=True
    )

    input_path = None

    _qp = st.query_params
    if "lat" in _qp and "lon" in _qp:
        try:
            _qp_lat = float(_qp["lat"])
            _qp_lon = float(_qp["lon"])
            del _qp["lat"]
            del _qp["lon"]
            _qp_loc = {
                "center": [_qp_lon, _qp_lat],
                "bbox": [_qp_lon - 0.01, _qp_lat - 0.01, _qp_lon + 0.01, _qp_lat + 0.01],
            }
            st.session_state["location"] = _qp_loc
            if "input_path" in st.session_state and os.path.exists(str(st.session_state["input_path"])):
                input_path = st.session_state["input_path"]
                st.success(f"📍 Tam ekrandan konum alındı: {_qp_lat:.4f}°N, {_qp_lon:.4f}°E")
            else:
                with st.spinner(f"Uydu görüntüsü indiriliyor ({_qp_lat:.4f}, {_qp_lon:.4f})..."):
                    try:
                        _qp_tile = download_location_imagery(_qp_loc)
                        if _qp_tile:
                            st.session_state["input_path"] = _qp_tile
                            input_path = _qp_tile
                            st.success(f"📍 Tam ekrandan konum alındı: {_qp_lat:.4f}°N, {_qp_lon:.4f}°E")
                        else:
                            st.warning("Uydu görüntüsü indirilemedi")
                    except Exception as e:
                        st.error(f"İndirme hatası: {e}")
        except (ValueError, TypeError):
            pass

    if loc_method == "Şehir Seç":
        city = st.selectbox("Şehir", list(LOCATIONS.keys()))
        loc_data = LOCATIONS[city]
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[loc_data["center"][1], loc_data["center"][0]], zoom_start=10
        )
        folium.Marker(
            [loc_data["center"][1], loc_data["center"][0]],
            popup=city,
            icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        ).add_to(m)
        st_folium(m, height=400, use_container_width=True, returned_objects=[])
        if st.button("Bu konumu kullan", key="use_city"):
            st.session_state["location"] = loc_data
            if "input_path" in st.session_state and os.path.exists(str(st.session_state["input_path"])):
                input_path = st.session_state["input_path"]
                st.success(f"Konum: {city} — mevcut veri kullanılıyor")
            else:
                with st.spinner("Uydu görüntüsü indiriliyor..."):
                    try:
                        tile_path = download_location_imagery(loc_data)
                        if tile_path:
                            st.session_state["input_path"] = tile_path
                            input_path = tile_path
                            st.success(f"Konum: {city} — uydu görüntüsü indirildi")
                        else:
                            st.warning("Uydu görüntüsü indirilemedi")
                    except Exception as e:
                        st.error(f"İndirme hatası: {e}")

    elif loc_method == "Koordinat Gir":
        col1, col2 = st.columns(2)
        with col1:
            lon = st.number_input("Boylam (Lon)", value=29.8239, format="%.4f")
        with col2:
            lat = st.number_input("Enlem (Lat)", value=40.7593, format="%.4f")
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker(
            [lat, lon],
            popup=f"{lat:.4f}°N, {lon:.4f}°E",
            icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        ).add_to(m)
        st_folium(m, height=400, use_container_width=True, returned_objects=[])
        if st.button("Bu konumu kullan", key="use_coords"):
            loc_data = {
                "center": [lon, lat],
                "bbox": [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
            }
            st.session_state["location"] = loc_data
            if "input_path" in st.session_state and os.path.exists(str(st.session_state["input_path"])):
                input_path = st.session_state["input_path"]
                st.success(f"Konum: {lat:.4f}, {lon:.4f} — mevcut veri kullanılıyor")
            else:
                with st.spinner("Uydu görüntüsü indiriliyor..."):
                    try:
                        tile_path = download_location_imagery(loc_data)
                        if tile_path:
                            st.session_state["input_path"] = tile_path
                            input_path = tile_path
                            st.success(f"Konum: {lat:.4f}, {lon:.4f} — uydu görüntüsü indirildi")
                        else:
                            st.warning("Uydu görüntüsü indirilemedi")
                    except Exception as e:
                        st.error(f"İndirme hatası: {e}")

    elif loc_method == "Haritadan Seç":
        import folium
        from streamlit_folium import st_folium

        st.markdown("**Haritaya tıklayarak konum seçin**")

        sel_lat = st.session_state.get("click_lat", None)
        sel_lon = st.session_state.get("click_lon", None)

        map_center = [sel_lat, sel_lon] if sel_lat else [40.7593, 29.8239]
        m = folium.Map(location=map_center, zoom_start=5 if sel_lat is None else 11)

        if sel_lat is not None and sel_lon is not None:
            folium.Marker(
                [sel_lat, sel_lon],
                popup=f"{sel_lat:.4f}°N, {sel_lon:.4f}°E",
                icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
            ).add_to(m)

        map_data = st_folium(m, height=400, use_container_width=True)

        if map_data and map_data.get("last_clicked"):
            st.session_state["click_lat"] = map_data["last_clicked"]["lat"]
            st.session_state["click_lon"] = map_data["last_clicked"]["lng"]
            st.rerun()

        sel_lat = st.session_state.get("click_lat", None)
        sel_lon = st.session_state.get("click_lon", None)

        if sel_lat is not None and sel_lon is not None:
            st.markdown(f"**Seçilen Konum:** `{sel_lat:.4f}°N, {sel_lon:.4f}°E`")
        else:
            st.info("Haritaya tıklayarak bir konum seçin")

        if st.button("Bu konumu kullan", key="use_map", disabled=(sel_lat is None)):
            loc_data = {
                "center": [sel_lon, sel_lat],
                "bbox": [
                    sel_lon - 0.01,
                    sel_lat - 0.01,
                    sel_lon + 0.01,
                    sel_lat + 0.01,
                ],
            }
            st.session_state["location"] = loc_data
            if "input_path" in st.session_state and os.path.exists(str(st.session_state["input_path"])):
                input_path = st.session_state["input_path"]
                st.success(f"Konum kaydedildi: {sel_lat:.4f}, {sel_lon:.4f} — mevcut veri kullanılıyor")
            else:
                with st.spinner("Uydu görüntüsü indiriliyor..."):
                    try:
                        tile_path = download_location_imagery(loc_data)
                        if tile_path:
                            st.session_state["input_path"] = tile_path
                            input_path = tile_path
                            st.success(f"Konum kaydedildi: {sel_lat:.4f}, {sel_lon:.4f}")
                        else:
                            st.warning("Uydu görüntüsü indirilemedi")
                    except Exception as e:
                        st.error(f"İndirme hatası: {e}")

    if "input_path" in st.session_state and input_path is None:
        input_path = st.session_state["input_path"]

    st.divider()
    st.subheader("🗺️ Tam Ekran Harita")
    st.caption("Ayrıntılı OpenStreetMap'te gezin, konum seçin")
    if st.button("🖥️ Tam Ekran Haritayı Aç", key="open_fullscreen_osm", use_container_width=True):
        import webbrowser, threading, http.server, socketserver

        _st_base_url = "http://localhost:8501"
        try:
            from streamlit.web.server.server import Server
            _srv = Server.get_current()
            _st_base_url = f"http://localhost:{_srv._port}"
        except Exception:
            try:
                _st_base_url = f"http://localhost:{os.environ.get('STREAMLIT_SERVER_PORT', '8501')}"
            except Exception:
                pass

        _osm_html = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>OpenStreetMap — Tam Ekran</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body,#map{width:100%;height:100%;font-family:system-ui,-apple-system,sans-serif}
#search-box{position:absolute;top:12px;left:60px;z-index:1000;width:340px;display:flex;gap:6px}
#search-box input{flex:1;padding:10px 14px;border:none;border-radius:8px;font-size:15px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
#search-box button{padding:10px 16px;border:none;border-radius:8px;background:#2563eb;color:#fff;font-size:14px;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.25);white-space:nowrap}
#search-box button:hover{background:#1d4ed8}
#coord-panel{position:absolute;bottom:56px;left:50%;transform:translateX(-50%);z-index:1000;background:#fff;border-radius:12px;padding:14px 22px;box-shadow:0 4px 16px rgba(0,0,0,.2);display:none;text-align:center;min-width:320px}
#coord-panel h3{margin:0 0 6px 0;font-size:15px;color:#374151}
#coord-panel .coords{font-size:20px;font-weight:700;color:#1d4ed8;margin:4px 0}
#coord-panel .addr{font-size:13px;color:#6b7280;margin-bottom:10px}
#coord-panel button{padding:8px 20px;border:none;border-radius:6px;background:#059669;color:#fff;font-size:14px;cursor:pointer;margin:2px}
#coord-panel button:hover{background:#047857}
#coord-panel button.copy-btn{background:#6366f1}
#coord-panel button.copy-btn:hover{background:#4f46e5}
.layer-btn{position:absolute;top:12px;right:12px;z-index:1000;display:flex;flex-direction:column;gap:4px}
.layer-btn button{padding:8px 12px;border:none;border-radius:6px;background:#fff;font-size:12px;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,.15)}
.layer-btn button.active{background:#2563eb;color:#fff}
.leaflet-control-zoom{margin-top:60px!important}
#toast{position:absolute;top:70px;left:50%;transform:translateX(-50%);z-index:2000;background:#059669;color:#fff;padding:10px 20px;border-radius:8px;display:none;font-size:14px}
#info-bar{position:absolute;bottom:28px;right:12px;z-index:1000;background:rgba(15,23,42,.92);color:#e2e8f0;font-size:16px;padding:20px 26px;border-radius:14px;display:flex;flex-direction:column;gap:9px;font-family:'SF Mono',SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;backdrop-filter:blur(12px);border:1px solid rgba(99,102,241,.25);box-shadow:0 4px 24px rgba(0,0,0,.4);min-width:380px}
#info-bar .row{display:flex;align-items:center;gap:12px}
#info-bar .icon{width:28px;text-align:center;font-size:18px;flex-shrink:0}
#info-bar .label{color:#818cf8;font-size:12px;text-transform:uppercase;letter-spacing:.6px;width:54px;flex-shrink:0}
#info-bar .val{color:#f1f5f9;font-weight:600;font-size:15px;letter-spacing:.3px}
#info-bar .divider-line{height:1px;background:rgba(99,102,241,.2);margin:2px 0}
</style>
</head>
<body>
<div id="map"></div>
<div id="search-box">
  <input id="q" placeholder="Konum ara... (ör: İstanbul, Ankara)" onkeydown="if(event.key==='Enter')doSearch()"/>
  <button onclick="doSearch()">Ara</button>
</div>
<div class="layer-btn">
  <button class="active" onclick="setLayer('osm',this)">OpenStreetMap</button>
  <button onclick="setLayer('sat',this)">Uydu</button>
  <button onclick="setLayer('topo',this)">Topografik</button>
  <button onclick="setLayer('dark',this)">Karanlık</button>
</div>
<div id="coord-panel">
  <h3>📍 Seçilen Konum</h3>
  <div class="coords" id="coord-text">—</div>
  <div class="addr" id="addr-text"></div>
  <button class="copy-btn" onclick="copyCoords()">📋 Koordinatları Kopyala</button>
  <button onclick="copyAndClose()">✅ Koordinatları Gir (Sidebar)</button>
</div>
<div id="toast">Kopyalandı!</div>
<div id="info-bar">
  <div class="row"><span class="icon">🎯</span><span class="label">MGRS</span><span class="val" id="mgrs-text">—</span></div>
  <div class="divider-line"></div>
  <div class="row"><span class="icon">🗺️</span><span class="label">UTM</span><span class="val" id="utm-text">—</span></div>
  <div class="divider-line"></div>
  <div class="row"><span class="icon">📏</span><span class="label">Ölçek</span><span class="val" id="scale-text">—</span></div>
  <div class="divider-line"></div>
  <div class="row"><span class="icon">📍</span><span class="label">İmleç</span><span class="val" id="cursor-text">—</span></div>
</div>
<script>
var _BANDS='CDEFGHJKLMNPQRSTUVWX';
var _COL_LETTERS='STUVWXYZABCDEFGHJKLMNPQRS';
var _ROW_LETTERS_N='ABCDEFGHJKLMNPQRSTUV';
var _ROW_LETTERS_S='FGHJKLMNPQRSTUVABCDE';

function _llToUtm(lat,lon){
  var a=6378137,f=1/298.257223563,e2=2*f-f*f,ep2=e2/(1-e2),e1=(1-Math.sqrt(1-e2))/(1+Math.sqrt(1-e2));
  var zn=Math.floor((lon+180)/6)+1,zs=lat>=0?'N':'S';
  var cm=(zn-1)*6-180+3;
  var latR=lat*Math.PI/180,sn=Math.sin(latR),cs=Math.cos(latR),tn=Math.tan(latR);
  var nu=a/Math.sqrt(1-e2*sn*sn),N=a/Math.sqrt(1-e2*sn*sn);
  var T=tn*tn,C=ep2*cs*cs,A=(lon-cm)*Math.PI/180;
  var M=a*((1-e2/4-3*e2*e2/64-5*e2*e2*e2/256)*latR-(3*e2/8+3*e2*e2/32+45*e2*e2*e2/1024)*Math.sin(2*latR)+(15*e2*e2/256+45*e2*e2*e2/1024)*Math.sin(4*latR)-(35*e2*e2*e2/3072)*Math.sin(6*latR));
  var k0=0.9996,E=k0*N*(A+(1-T+C)*A*A*A/6+(5-18*T+T*T+72*C-58*ep2)*A*A*A*A*A/120)+500000;
  var nn=k0*(M+N*Math.tan(latR)*(A*A/2+(5-T+9*C+4*C*C-19*ep2)*A*A*A*A/24+(61-58*T+T*T+600*C-220*ep2)*A*A*A*A*A*A/720));
  if(lat<0)nn+=10000000;
  return{zone:zn,letter:zs,easting:E,northing:nn}
}

function _utmToMgrs(zn,zs,E,N){
  var bi=Math.round((E-1)/100000)-1,cL=_COL_LETTERS[(zn-1)%3*8+bi];
  var bset=zn%2===0?1:0;
  var rI=Math.floor(N%2000000/100000);
  var rL=(zs==='N'?_ROW_LETTERS_N:_ROW_LETTERS_S)[(bset*5+rI)%20];
  var e1k=Math.floor(E%100000),n1k=Math.floor(N%100000);
  return String(zn)+zs+' '+cL+rL+' '+('00000'+e1k).slice(-5)+' '+('00000'+n1k).slice(-5);
}

function _getScale(zoom,lat){
  var c=2*Math.PI*6378137;
  var ms=c*Math.cos(lat*Math.PI/180)/Math.pow(2,zoom+8);
  var nice=[100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,5000000,10000000];
  var best=nice[0];for(var i=0;i<nice.length;i++){if(nice[i]<ms*150)best=nice[i]}
  return '1:'+best.toLocaleString();
}

var map=L.map('map',{zoomControl:true,attributionControl:true}).setView([41.05,29.0],6);
var layers={
  osm:L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}),
  sat:L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'© Esri'}),
  topo:L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{maxZoom:17,attribution:'© OpenTopoMap'}),
  dark:L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',{maxZoom:19,attribution:'© CartoDB'})
};
var currentLayer=layers.osm.addTo(map);
var marker=null;
var selectedLat=null,selectedLon=null;

function _updateInfoBar(lat,lon){
  try{
    var u=_llToUtm(lat,lon);
    document.getElementById('mgrs-text').textContent=_utmToMgrs(u.zone,u.letter,u.easting,u.northing);
    document.getElementById('utm-text').textContent=u.zone+u.letter+' '+Math.round(u.easting)+'E '+Math.round(u.northing)+'N';
    document.getElementById('scale-text').textContent=_getScale(map.getZoom(),lat);
    document.getElementById('cursor-text').textContent=lat.toFixed(6)+'°N, '+lon.toFixed(6)+'°E';
  }catch(e){}
}

map.on('mousemove',function(e){_updateInfoBar(e.latlng.lat,e.latlng.lng)});
map.on('zoomend moveend',function(){
  var c=map.getCenter();_updateInfoBar(c.lat,c.lng);
});

L.control.scale({metric:true,imperial:false,position:'bottomleft'}).addTo(map);

function setLayer(name,btn){
  map.removeLayer(currentLayer);
  currentLayer=layers[name].addTo(map);
  document.querySelectorAll('.layer-btn button').forEach(function(b){b.classList.remove('active')});
  btn.classList.add('active');
}

map.on('click',function(e){
  selectedLat=e.latlng.lat;
  selectedLon=e.latlng.lng;
  if(marker)map.removeLayer(marker);
  marker=L.marker(e.latlng,{draggable:true}).addTo(map);
  marker.on('dragend',function(ev){
    selectedLat=ev.target.getLatLng().lat;
    selectedLon=ev.target.getLatLng().lng;
    updatePanel();
  });
  updatePanel();
  reverseGeocode(selectedLat,selectedLon);
});

function updatePanel(){
  var p=document.getElementById('coord-panel');
  p.style.display='block';
  document.getElementById('coord-text').textContent=selectedLat.toFixed(6)+'°N, '+selectedLon.toFixed(6)+'°E';
}

function reverseGeocode(lat,lon){
  fetch('https://nominatim.openstreetmap.org/reverse?format=json&lat='+lat+'&lon='+lon+'&accept-language=tr')
    .then(function(r){return r.json()})
    .then(function(d){
      var addr=d.display_name||'';
      document.getElementById('addr-text').textContent=addr;
    })
    .catch(function(){document.getElementById('addr-text').textContent='';});
}

function doSearch(){
  var q=document.getElementById('q').value.trim();
  if(!q)return;
  fetch('https://nominatim.openstreetmap.org/search?format=json&q='+encodeURIComponent(q)+'&limit=1&accept-language=tr')
    .then(function(r){return r.json()})
    .then(function(d){
      if(d.length>0){
        var r=d[0];
        var lat=parseFloat(r.lat),lon=parseFloat(r.lon);
        map.setView([lat,lon],14);
        if(marker)map.removeLayer(marker);
        selectedLat=lat;selectedLon=lon;
        marker=L.marker([lat,lon],{draggable:true}).addTo(map);
        marker.on('dragend',function(ev){
          selectedLat=ev.target.getLatLng().lat;
          selectedLon=ev.target.getLatLng().lng;
          updatePanel();reverseGeocode(selectedLat,selectedLon);
        });
        updatePanel();
        document.getElementById('addr-text').textContent=r.display_name||'';
      }else{alert('Sonuç bulunamadı');}
    });
}

function copyCoords(){
  var t=selectedLat.toFixed(6)+', '+selectedLon.toFixed(6);
  navigator.clipboard.writeText(t);
  var toast=document.getElementById('toast');toast.style.display='block';
  setTimeout(function(){toast.style.display='none'},1500);
}

function copyAndClose(){
  window.location.href='__ST_BASE_URL__/?lat='+selectedLat.toFixed(6)+'&lon='+selectedLon.toFixed(6);
}
</script>
</body>
</html>"""

        _osm_html = _osm_html.replace("__ST_BASE_URL__", _st_base_url)
        _osm_dir = Path(st.session_state.get("out_dir", tempfile.mkdtemp()))
        _osm_path = str(_osm_dir / "fullscreen_osm.html")
        with open(_osm_path, "w", encoding="utf-8") as _f:
            _f.write(_osm_html)

        class _OSMHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(_osm_dir), **kw)
            def log_message(self, *a):
                pass

        if "osm_http_port" not in st.session_state:
            _httpd = socketserver.TCPServer(("", 0), _OSMHandler)
            _port = _httpd.server_address[1]
            st.session_state["osm_http_port"] = _port
            threading.Thread(target=_httpd.serve_forever, daemon=True).start()
        else:
            _port = st.session_state["osm_http_port"]

        webbrowser.open(f"http://localhost:{_port}/fullscreen_osm.html")
        st.success(f"Harita açıldı: http://localhost:{_port}/fullscreen_osm.html")

    # Dosya bilgisi
    if input_path and os.path.exists(str(input_path)):
        with st.expander("📊 Veri Bilgisi"):
            info = get_image_info(str(input_path))
            for k, v in info.items():
                st.write(f"**{k}:** {v}")

    st.divider()
    out_dir = tempfile.mkdtemp()
    st.session_state["out_dir"] = out_dir


# ── MAIN CONTENT ─────────────────────────────────────────────────────
st.title(demo_choice)
st.markdown(DEMOS[demo_choice])

# Output placeholder
result_placeholder = st.empty()

# ── DEMO 01: Söyle, Ayır ────────────────────────────────────────────
if demo_choice.startswith("01"):
    prompt = st.text_input(
        "Ne arıyorsun?", value="road", placeholder="road, buildings, trees, water..."
    )
    

    if st.button("🚀 Çalıştır", type="primary"):
        if not input_path:
            st.warning("Önce sidebar'dan veri yükleyin veya konum seçin")
        else:
            import gc, torch, rasterio
            from rasterio.plot import show as rioshow

            out = Path(st.session_state["out_dir"])
            results = {}

            # CLIPSeg
            with st.spinner("CLIPSeg çalışıyor..."):
                try:
                    from geoai import CLIPSegmentation

                    model = CLIPSegmentation()
                    clip_path = str(out / "clipseg.tif")
                    model.segment_image(
                        input_path=str(input_path),
                        output_path=clip_path,
                        text_prompt=prompt,
                        threshold=0.5,
                        smoothing_sigma=1.0,
                    )
                    results["CLIPSeg"] = clip_path
                except Exception as e:
                    st.error(f"CLIPSeg: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # GroundedSAM
            with st.spinner("GroundedSAM çalışıyor..."):
                try:
                    from geoai.segment import GroundedSAM

                    torch.cuda.empty_cache()
                    gc.collect()
                    model = GroundedSAM(
                        detector_id="IDEA-Research/grounding-dino-tiny",
                        segmenter_id="facebook/sam-vit-base",
                        threshold=0.35,
                    )

                    _orig_gsam_detect = model._detect
                    def _gsam_detect_filtered(image, labels, _orig=_orig_gsam_detect, _max_ratio=0.35):
                        results = _orig(image, labels)
                        if results:
                            img_w, img_h = image.size
                            img_area = img_w * img_h
                            return [
                                r for r in results
                                if (r.box.xmax - r.box.xmin) * (r.box.ymax - r.box.ymin) / img_area <= _max_ratio
                            ]
                        return results
                    model._detect = _gsam_detect_filtered

                    gsam_path = str(out / "grounded_sam.tif")
                    result = model.segment_image(
                        input_path=str(input_path),
                        output_path=gsam_path,
                        text_prompts=prompt,
                        export_polygons=True,
                        smoothing_sigma=1.0,
                        polygon_refinement=True,
                    )
                    results["GroundedSAM"] = gsam_path
                except Exception as e:
                    st.error(f"GroundedSAM: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # SamGeo
            with st.spinner("SamGeo çalışıyor..."):
                try:
                    from geoai.sam import SamGeo

                    torch.cuda.empty_cache()
                    gc.collect()
                    sam = SamGeo(model="facebook/sam-vit-huge", automatic=True)
                    sam_path = str(out / "samgeo.tif")
                    sam.generate(
                        str(input_path), output=sam_path, foreground=True, unique=True
                    )
                    results["SamGeo"] = sam_path
                except Exception as e:
                    st.error(f"SamGeo: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            # xView2 Loc (Bina Segmentasyonu)
            with st.spinner("xView2 Loc (bina tespiti) çalışıyor..."):
                try:
                    import cv2
                    XVIEW2_DIR = Path("/home/openzeka/Desktop/mekansal-veri/xView2-deploy")
                    sys.path.insert(0, str(XVIEW2_DIR))
                    from models import XViewFirstPlaceLocModel

                    torch.cuda.empty_cache()
                    gc.collect()
                    dp_mode = torch.cuda.device_count() <= 1
                    loc_model = XViewFirstPlaceLocModel("34", models_folder=str(XVIEW2_DIR / "weights"), dp_mode=dp_mode)

                    with rasterio.open(str(input_path)) as src:
                        img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                        if img.dtype != np.uint8:
                            p2, p98 = np.nanpercentile(img[img > 0], [2, 98])
                            img = np.clip((img - p2) / (p98 - p2 + 1e-10) * 255, 0, 255).astype(np.uint8)
                        h, w = img.shape[:2]
                        target_h = ((h - 1) // 256 + 1) * 256
                        target_w = ((w - 1) // 256 + 1) * 256
                        if h != target_h or w != target_w:
                            padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                            padded[:h, :w] = img
                            img = padded

                    def preprocess_inputs(x):
                        x = np.asarray(x, dtype='float32')
                        x /= 127
                        x -= 1
                        return x

                    inp = preprocess_inputs(img.copy())
                    inp_list = [inp, inp[::-1, ...], inp[:, ::-1, ...], inp[::-1, ::-1, ...]]
                    inp_arr = np.asarray(inp_list, dtype='float')
                    inp_tensor = torch.from_numpy(inp_arr.transpose((0, 3, 1, 2))).float().unsqueeze(0)

                    with torch.no_grad():
                        loc_out = loc_model.forward(inp_tensor)
                    loc_mask = loc_out.numpy().squeeze()[:h, :w]

                    xv2_path = str(out / "xview2_loc.tif")
                    with rasterio.open(str(input_path)) as src:
                        profile = src.profile.copy()
                        profile.update(count=1, dtype='uint8')
                        with rasterio.open(xv2_path, 'w', **profile) as dst:
                            dst.write(loc_mask, 1)

                    results["xView2-Loc"] = xv2_path
                    del loc_model
                except Exception as e:
                    st.error(f"xView2-Loc: {e}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            if results:
                cols = st.columns(len(results) + 1)
                with cols[0]:
                    with rasterio.open(str(input_path)) as src:
                        if src.count >= 3:
                            rgb = _norm_rgb(
                                np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                            )
                            st.image(rgb, caption="Orijinal", use_container_width=True)
                colors = {
                    "CLIPSeg": "Greens",
                    "GroundedSAM": "autumn",
                    "SamGeo": "cool",
                    "xView2-Loc": "Blues",
                }
                for i, (name, mask_path) in enumerate(results.items()):
                    with cols[i + 1]:
                        with rasterio.open(mask_path) as msrc:
                            mask = msrc.read(1).astype(np.float32)
                            mask = np.where(mask == 0, 0, 255).astype(np.uint8)
                        if name == "xView2-Loc":
                            caption = f"{name} (bina tespiti)"
                        elif name == "SamGeo":
                            caption = f"{name} (otomatik)"
                        else:
                            caption = f'{name}: "{prompt}"'
                        st.image(mask, caption=caption, use_container_width=True)
            else:
                st.warning("Hiçbir model çalışmadı.")


# ── DEMO 02: Fotoğrafa Sor ───────────────────────────────────────────
elif demo_choice.startswith("02"):
    if input_path and os.path.exists(str(input_path)):
        try:
            import rasterio
            with rasterio.open(str(input_path)) as src:
                if src.count >= 3:
                    _preview = _norm_rgb(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
                    _c1, _c2 = st.columns([1, 2])
                    with _c1:
                        st.image(_preview, caption="Seçilen Uydu Görüntüsü", use_container_width=True)
                else:
                    _band = src.read(1).astype(np.float32)
                    _valid = _band[_band > 0]
                    if len(_valid) > 0:
                        _p2, _p98 = np.nanpercentile(_valid, [2, 98])
                        _band = np.clip((_band - _p2) / (_p98 - _p2 + 1e-10) * 255, 0, 255)
                    _c1, _c2 = st.columns([1, 2])
                    with _c1:
                        st.image(_band.astype(np.uint8), caption="Seçilen Uydu Görüntüsü (tek bant)", use_container_width=True)
        except Exception:
            try:
                from PIL import Image as PILImage
                st.image(PILImage.open(str(input_path)), caption="Seçilen Uydu Görüntüsü", use_container_width=True)
            except Exception:
                st.warning("Görüntü önizlenemiyor")
    else:
        st.info("Önce sidebar'dan bir konum seçin.")

    default_qs = [
        "Bu uydu görüntüsünü detaylı şekilde tarif et. Yerleşim yerleri, yollar, su kaynakları ve yeşil alanlar dahil tüm önemli unsurları belirt.",
        "Görüntüde kaç bina tahmin ediyorsun? Yoğunluk olarak nasıl dağılmışlar? Kırsal mı kentsel mi bir alan?",
        "Görüntüde herhangi bir su kütlesi (göl, nehir, deniz, baraj) var mı? Varsa konumunu ve büyüklüğünü tahmin et.",
        "Bitki örtüsü hakkında ne söyleyebilirsin? Tarım alanı mı, doğal orman mı, yoksa park alanı mı? Yoğunluk seviyesi nedir?",
        "Arazi kullanımını sınıflandır: tarım, konut, sanayi, ulaşım, boş arazi. Her birin tahmini oranını ver.",
        "Bu bölgede olası afet riskleri nelerdir? Sel, heyelan, deprem açısından değerlendir.",
    ]
    questions_text = st.text_area(
        "Sorular (her satıra bir tane)", value="\n".join(default_qs), height=200
    )
    questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

    if st.button("🚀 Çalıştır", type="primary"):
        if not input_path:
            st.warning("Önce sidebar'dan veri yükleyin veya konum seçin")
        else:
            import requests as req
            import base64, io
            from PIL import Image as PILImage

            img = PILImage.open(str(input_path))
            if img.mode != "RGB":
                img = img.convert("RGB")
            if max(img.size) > 1024:
                ratio = 1024 / max(img.size)
                img = img.resize(
                    (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                    PILImage.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            for q in questions:
                with st.spinner(f'Soruluyor: "{q[:50]}..."'):
                    payload = {
                        "model": LLM_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "Sen uzman bir uydu görüntüsü analistisin. Verilen uydu görüntüsünü "
                                    "en ince ayrıntısına kadar analiz et. Cevapların her zaman uzun, "
                                    "detaylı ve yapılandırılmış olsun. Kısa veya eksik cevap verme. "
                                    "Gördüğün her unsuru say, konumlarını belirt, büyüklüklerini tahmin et. "
                                    "Emin olmadığın konularda 'tahminen' veya 'muhtemelen' ifadelerini kullan ama "
                                    "asla 'None' veya boş cevap verme. En az 3-4 cümle yaz."
                                ),
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{b64}"
                                        },
                                    },
                                    {"type": "text", "text": q},
                                ],
                            },
                        ],
                        "max_tokens": 2048,
                        "temperature": 0.4,
                    }
                    resp = req.post(
                        LLM_API_URL,
                        json=payload,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        _rj = resp.json()
                        answer = _rj["choices"][0]["message"]["content"]
                        _finish = _rj["choices"][0].get("finish_reason", "")
                        if answer is None and _finish == "length":
                            retry_payload = dict(payload)
                            retry_payload["max_tokens"] = 4096
                            retry_resp = req.post(LLM_API_URL, json=retry_payload, timeout=120)
                            if retry_resp.status_code == 200:
                                answer = retry_resp.json()["choices"][0]["message"]["content"]
                        if answer is None:
                            for _attempt in range(2):
                                retry_resp = req.post(LLM_API_URL, json=payload, timeout=120)
                                if retry_resp.status_code == 200:
                                    answer = retry_resp.json()["choices"][0]["message"]["content"]
                                    if answer is not None:
                                        break
                        if not answer:
                            answer = "Model bu soruya cevap üretemedi. Lütfen farklı bir şekilde sorun."
                        st.markdown(f"**Q:** {q}")
                        st.markdown(f"**A:** {answer}")
                    else:
                        st.error(f"API hatası: {resp.status_code}")


# ── DEMO 03: Agentic Analysis ────────────────────────────────────────
elif demo_choice.startswith("03"):
    task = st.text_area(
        "Görev",
        value="İstanbul Boğaziçi bölgesinin NDVI analizini yap",
        placeholder="Örn: Ankara'daki yeşil alan oranını hesapla, İzmir körfezi NDVI analizi...",
        height=80,
    )

    if st.button("🚀 Çalıştır", type="primary") and task:
        import gc, torch, json
        import requests as req

        API_URL = LLM_API_URL
        MODEL = LLM_MODEL
        out = Path(st.session_state["out_dir"])

        TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "geocode",
                    "description": "Bir yer adini koordinata cevir. Ornegin: Istanbul, Ankara, Paris, Tokyo. Sonuc olarak enlem/boylam doner.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_name": {
                                "type": "string",
                                "description": "Yer adi. Ornegin: Istanbul, Ankara Kizilay, Paris Eiffel Tower",
                            }
                        },
                        "required": ["location_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "download_satellite_imagery",
                    "description": "Verilen koordinatlar icin uydu goruntusu indirir. GeoTIFF olarak kaydeder.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "number",
                                "description": "Enlem (latitude)",
                            },
                            "lon": {
                                "type": "number",
                                "description": "Boylam (longitude)",
                            },
                            "zoom": {
                                "type": "integer",
                                "description": "Yaklasma seviyesi (12-18). Varsayilan: 18",
                            },
                        },
                        "required": ["lat", "lon"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "download_sample_data",
                    "description": "Ornek uydu verisi indir. Anahtarlar: sentinel2, naip, parking, trees, las_vegas_2019, las_vegas_2022",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Ornek veri anahtari",
                            }
                        },
                        "required": ["key"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_ndvi",
                    "description": "NDVI (Normalize Edilmis Bitki Orani) hesapla. -1 ile 1 arasinda deger doner. 0.3 ustu yesil alan, 0.6 ustu yogun bitki ortusu.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Goruntu yolu",
                            },
                            "red_band": {
                                "type": "integer",
                                "description": "Kirmizi bant (varsayilan: 1)",
                            },
                            "nir_band": {
                                "type": "integer",
                                "description": "NIR bant (varsayilan: 4)",
                            },
                        },
                        "required": ["image_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "detect_objects",
                    "description": "Metin ile nesne tespiti. CLIPSeg kullanir.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Goruntu yolu",
                            },
                            "prompt": {"type": "string", "description": "Nesne adi (road, buildings, trees, water...)"},
                        },
                        "required": ["image_path", "prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_vlm",
                    "description": "Uydu goruntusune soru sor. VLM ile gorsel analiz yapar.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Goruntu yolu",
                            },
                            "question": {"type": "string", "description": "Soru"},
                        },
                        "required": ["image_path", "question"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen mekansal veri analiz ajansin. Kullanici bir yer adi veya gorev verdiginde:\n"
                    "1. Oncelikle 'geocode' ile yerin koordinatlarini bul.\n"
                    "2. Ardindan 'download_satellite_imagery' ile o konumun uydu goruntusunu indir.\n"
                    "3. Sonra istenen analizi yap (NDVI, nesne tespiti, VLM analizi vs.).\n"
                    "4. Sonuclari Turkce ve detayli acikla.\n"
                    "NDVI icin: 3 bantli goruntulerde red=b1, nir=b1 kullanilir (tek bant). "
                    "4+ bantli Sentinel-2 verisinde red=b1, nir=b4 kullanilir.\n"
                    "Ornek veri anahtarlari: sentinel2, naip, parking, trees, las_vegas_2019, las_vegas_2022"
                ),
            },
            {"role": "user", "content": task},
        ]

        log_area = st.empty()
        log_lines = []
        step = 0

        for _ in range(15):
            payload = {
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": 2048,
                "temperature": 0.3,
            }
            resp = req.post(API_URL, json=payload, timeout=120)
            if resp.status_code != 200:
                log_lines.append(f"❌ API hatası: {resp.status_code}")
                break

            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            finish = choice.get("finish_reason", "")

            if msg.get("content"):
                log_lines.append(f"🤖 {msg['content'][:500]}")

            if finish == "stop" or not msg.get("tool_calls"):
                break

            messages.append(msg)

            for tc in msg["tool_calls"]:
                step += 1
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                log_lines.append(
                    f"**Adım {step}:** `{fn_name}({json.dumps(fn_args, ensure_ascii=False)[:100]})`"
                )

                result = {
                    "tool": fn_name,
                    "args": fn_args,
                    "status": "ok",
                    "output": "",
                }

                if fn_name == "geocode":
                    loc_name = fn_args.get("location_name", "")
                    try:
                        coords = geocode_location(loc_name)
                        if coords:
                            lon, lat = coords
                            result["output"] = json.dumps(
                                {"lat": lat, "lon": lon, "name": loc_name}
                            )
                            log_lines.append(f"  ✅ {loc_name}: {lat:.4f}°N, {lon:.4f}°E")
                        else:
                            result["status"] = "error"
                            result["output"] = f"Konum bulunamadi: {loc_name}"
                            log_lines.append(f"  ❌ Konum bulunamadi: {loc_name}")
                    except Exception as e:
                        result["status"] = "error"
                        result["output"] = str(e)
                        log_lines.append(f"  ❌ {e}")

                elif fn_name == "download_satellite_imagery":
                    lat = fn_args.get("lat", 41.0)
                    lon = fn_args.get("lon", 29.0)
                    zoom = fn_args.get("zoom", 18)
                    year = fn_args.get("year", None)
                    try:
                        loc_data = {
                            "center": [lon, lat],
                            "bbox": [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01],
                        }
                        path = download_location_imagery(loc_data, zoom=zoom, year=year)
                        if path:
                            result["output"] = path
                            result["path"] = path
                            log_lines.append(f"  ✅ Uydu görüntüsü indirildi: {Path(path).name}")
                        else:
                            result["status"] = "error"
                            result["output"] = "Uydu goruntusu indirilemedi"
                            log_lines.append(f"  ❌ Uydu goruntusu indirilemedi")
                    except Exception as e:
                        result["status"] = "error"
                        result["output"] = str(e)
                        log_lines.append(f"  ❌ {e}")

                elif fn_name == "download_sample_data":
                    key = fn_args.get("key", "")
                    try:
                        import geoai
                        url = None
                        for k, v in SAMPLE_DATA.items():
                            if key.lower().replace("_", " ") in k.lower() or key.lower() in k.lower() or k.lower() in key.lower():
                                url = v
                                break
                        if not url:
                            url = SAMPLE_DATA.get(key, None)
                        if not url:
                            available = ", ".join(SAMPLE_DATA.keys())
                            result["status"] = "error"
                            result["output"] = f"Bilinmeyen anahtar: {key}. Mevcut: {available}"
                            log_lines.append(f"  ❌ Bilinmeyen anahtar: {key}")
                            continue
                        path = geoai.download_file(url)
                        result["output"] = path
                        result["path"] = path
                        log_lines.append(f"  ✅ İndirildi: {Path(path).name}")
                    except Exception as e:
                        result["status"] = "error"
                        result["output"] = str(e)
                        log_lines.append(f"  ❌ {e}")

                elif fn_name == "calculate_ndvi":
                    import rasterio
                    img_path = fn_args.get("image_path", "")
                    red_b = fn_args.get("red_band", 1)
                    nir_b = fn_args.get("nir_band", 4)
                    try:
                        with rasterio.open(img_path) as src:
                            nb = src.count
                            red = src.read(min(red_b, nb)).astype(np.float32)
                            nir = src.read(min(nir_b, nb)).astype(np.float32)
                            red = np.where(red == 0, np.nan, red)
                            nir = np.where(nir == 0, np.nan, nir)
                            ndvi = np.clip((nir - red) / (nir + red + 1e-10), -1, 1)
                            valid = ndvi[~np.isnan(ndvi)]
                            if len(valid) == 0:
                                result["status"] = "error"
                                result["output"] = "Gecerli piksel bulunamadi"
                                log_lines.append(f"  ❌ Gecerli piksel yok")
                            else:
                                stats = {
                                    "mean": float(np.mean(valid)),
                                    "min": float(np.min(valid)),
                                    "max": float(np.max(valid)),
                                    "vegetation_pct": float(
                                        np.sum(valid > 0.3) / len(valid) * 100
                                    ),
                                    "dense_vegetation_pct": float(
                                        np.sum(valid > 0.6) / len(valid) * 100
                                    ),
                                }
                                result["output"] = json.dumps(stats)
                                log_lines.append(
                                    f"  ✅ NDVI: ort={stats['mean']:.2f}, bitki=%{stats['vegetation_pct']:.1f}"
                                )

                                ndvi_path = str(out / "ndvi.tif")
                                meta = src.meta.copy()
                                meta.update(
                                    {"count": 1, "dtype": "float32", "nodata": np.nan}
                                )
                                with rasterio.open(ndvi_path, "w", **meta) as dst:
                                    dst.write(ndvi, 1)

                                import matplotlib
                                matplotlib.use("Agg")
                                import matplotlib.pyplot as plt

                                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                                if nb >= 3:
                                    from rasterio.plot import show as rioshow
                                    rioshow(
                                        src.read([1, 2, 3]),
                                        transform=src.transform,
                                        ax=axes[0],
                                    )
                                    axes[0].set_title("Orijinal")
                                im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
                                axes[1].set_title("NDVI")
                                plt.colorbar(im, ax=axes[1], fraction=0.046, label="NDVI")
                                plt.tight_layout()
                                png_path = str(out / "ndvi.png")
                                plt.savefig(
                                    png_path,
                                    dpi=150,
                                    bbox_inches="tight",
                                    facecolor="white",
                                )
                                plt.close()
                                st.image(
                                    png_path,
                                    caption="NDVI Analizi",
                                    use_container_width=True,
                                )
                    except Exception as e:
                        result["status"] = "error"
                        result["output"] = str(e)
                        log_lines.append(f"  ❌ {e}")

                elif fn_name == "detect_objects":
                    import rasterio
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()
                        from geoai import CLIPSegmentation

                        m = CLIPSegmentation()
                        mp = str(out / f"seg_{fn_args.get('prompt', 'obj')}.tif")
                        m.segment_image(
                            input_path=fn_args["image_path"],
                            output_path=mp,
                            text_prompt=fn_args["prompt"],
                            threshold=0.5,
                            smoothing_sigma=1.0,
                        )
                        with rasterio.open(mp) as ms:
                            coverage = (
                                np.sum(ms.read(1) > 0) / (ms.width * ms.height) * 100
                            )
                        result["output"] = json.dumps(
                            {
                                "prompt": fn_args["prompt"],
                                "coverage_pct": round(coverage, 2),
                            }
                        )
                        result["mask_path"] = mp
                        log_lines.append(f"  ✅ %{coverage:.1f} coverage")
                        del m
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as e:
                        result["status"] = "error"
                        result["output"] = str(e)
                        log_lines.append(f"  ❌ {e}")

                elif fn_name == "ask_vlm":
                    try:
                        import base64, io
                        from PIL import Image as PILImage
                        import rasterio

                        img_path = fn_args["image_path"]
                        try:
                            with rasterio.open(img_path) as src:
                                if src.count >= 3:
                                    arr = _norm_rgb(np.transpose(src.read([1, 2, 3]), (1, 2, 0)))
                                else:
                                    band = src.read(1).astype(np.float32)
                                    valid = band[band > 0]
                                    if len(valid) > 0:
                                        p2, p98 = np.nanpercentile(valid, [2, 98])
                                        band = np.clip((band - p2) / (p98 - p2 + 1e-10) * 255, 0, 255)
                                    else:
                                        band = np.zeros_like(band)
                                    arr = np.stack([band.astype(np.uint8)] * 3, axis=2)
                            img = PILImage.fromarray(arr)
                        except Exception:
                            img = PILImage.open(img_path)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        if max(img.size) > 1024:
                            r = 1024 / max(img.size)
                            img = img.resize(
                                (int(img.size[0] * r), int(img.size[1] * r)),
                                PILImage.LANCZOS,
                            )
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        vlm_resp = req.post(
                            API_URL,
                            json={
                                "model": MODEL,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{b64}"
                                                },
                                            },
                                            {
                                                "type": "text",
                                                "text": fn_args["question"],
                                            },
                                        ],
                                    }
                                ],
                                "max_tokens": 1024,
                                "temperature": 0.3,
                            },
                            timeout=120,
                        )
                        if vlm_resp.status_code == 200:
                            ans = vlm_resp.json()["choices"][0]["message"]["content"]
                            result["output"] = ans
                            log_lines.append(f"  ✅ {ans[:200]}")
                        else:
                            result["status"] = "error"
                            result["output"] = f"API: {vlm_resp.status_code}"
                    except Exception as e:
                        result["status"] = "error"
                        result["output"] = str(e)
                        log_lines.append(f"  ❌ {e}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

            log_area.markdown("\n\n".join(log_lines))

        log_area.markdown("\n\n".join(log_lines))
        st.success(f"Ajan tamamlandı ({step} adım)")


# ── DEMO 04: xView2 Hasar Tespiti ─────────────────────────────────────────
elif demo_choice.startswith("04"):
    XVIEW2_DIR = Path("/home/openzeka/Desktop/mekansal-veri/xView2-deploy")
    CHIPS_PRE = XVIEW2_DIR / "tests" / "data" / "output" / "chips" / "pre"
    CHIPS_POST = XVIEW2_DIR / "tests" / "data" / "output" / "chips" / "post"
    TILE_PRE = XVIEW2_DIR / "tests" / "data" / "input" / "pre"
    TILE_POST = XVIEW2_DIR / "tests" / "data" / "input" / "post"

    st.markdown("""
    **Bina Hasar Tespiti** — Afet sonrası uydu görüntülerinden bina hasarını sınıflandırır.
    
    Hasar sınıfları: 🟢 Hasarsız · 🟡 Hafif Hasar · 🟠 Ağır Hasar · 🔴 Yıkılmış
    """)

    use_test_data = st.checkbox("Örnek veriyi kullan", value=True)

    pre_path = None
    post_path = None

    if use_test_data:
        if CHIPS_PRE.exists() and CHIPS_POST.exists():
            chip_pre_files = sorted(CHIPS_PRE.glob("*.tif"))
            chip_post_files = sorted(CHIPS_POST.glob("*.tif"))
            if chip_pre_files and chip_post_files:
                chip_pairs = []
                for pf in chip_pre_files:
                    idx = pf.name.split("_")[0]
                    corresponding = CHIPS_POST / f"{idx}_post.tif"
                    if corresponding.exists():
                        chip_pairs.append((pf.name, corresponding.name, idx))
                if chip_pairs:
                    sel_idx = st.selectbox(
                        "Chip çifti seç",
                        range(len(chip_pairs)),
                        format_func=lambda i: f"Chip {chip_pairs[i][2]} (pre/post eşleşmeli)",
                    )
                    pre_name, post_name, _ = chip_pairs[sel_idx]
                    pre_path = str(CHIPS_PRE / pre_name)
                    post_path = str(CHIPS_POST / post_name)
                    st.info(f"Pre: `{pre_name}` | Post: `{post_name}`")
                else:
                    st.warning("Eşleşmiş chip çifti bulunamadı.")
                    use_test_data = False
            else:
                st.warning("Chip dosyaları bulunamadı.")
                use_test_data = False
        elif TILE_PRE.exists() and TILE_POST.exists():
            pre_files = sorted(TILE_PRE.glob("*.tif"))
            post_files = sorted(TILE_POST.glob("*.tif"))
            if pre_files and post_files:
                sel_pre = st.selectbox("Pre görüntü seç", range(len(pre_files)), format_func=lambda i: pre_files[i].name)
                sel_post = st.selectbox("Post görüntü seç", range(len(post_files)), format_func=lambda i: post_files[i].name)
                pre_path = str(pre_files[sel_pre])
                post_path = str(post_files[sel_post])
                st.info(f"Pre: `{pre_files[sel_pre].name}` | Post: `{post_files[sel_post].name}`")
                st.warning("Bu görüntüler farklı konumları kapsıyor olabilir — chip çiftleri tercih edilir.")
            else:
                st.warning("Test verisi bulunamadı.")
                use_test_data = False
        else:
            st.warning("Test verisi bulunamadı. Lütfen manuel olarak yükleyin.")
            use_test_data = False

    if not use_test_data:
        st.markdown("**Önceki (pre-disaster) görüntü:**")
        pre_uploaded = st.file_uploader("Pre görüntü yükle", type=["tif", "tiff", "png", "jpg"], key="xv2_pre")
        st.markdown("**Sonraki (post-disaster) görüntü:**")
        post_uploaded = st.file_uploader("Post görüntü yükle", type=["tif", "tiff", "png", "jpg"], key="xv2_post")

        if pre_uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(pre_uploaded.name).suffix)
            tmp.write(pre_uploaded.read())
            tmp.close()
            pre_path = tmp.name
        if post_uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(post_uploaded.name).suffix)
            tmp.write(post_uploaded.read())
            tmp.close()
            post_path = tmp.name

    model_size = st.selectbox("Model Boyutu", ["34", "50", "92", "154"], index=0,
                               help="Res34 en hızlı, SeNet154 en doğru (ama en yavaş)")

    if st.button("🚀 Hasar Tespiti Çalıştır", type="primary"):
        if not pre_path or not post_path:
            st.error("Hem pre hem post görüntü gerekli!")
        else:
            import gc, torch, cv2, rasterio
            sys.path.insert(0, str(XVIEW2_DIR))
            from models import XViewFirstPlaceLocModel, XViewFirstPlaceClsModel

            out = Path(st.session_state["out_dir"])

            with st.spinner("Model yükleniyor ve hasar tespiti yapılıyor..."):
                progress = st.progress(0, text="Loc modeli yükleniyor...")
                try:
                    dp_mode = torch.cuda.device_count() <= 1
                    loc_wrapper = XViewFirstPlaceLocModel(model_size, models_folder=str(XVIEW2_DIR / "weights"), dp_mode=dp_mode)
                    progress.progress(20, text="Loc modeli yüklendi. Cls modeli yükleniyor...")
                    cls_wrapper = XViewFirstPlaceClsModel(model_size, models_folder=str(XVIEW2_DIR / "weights"), dp_mode=dp_mode)
                    progress.progress(40, text="Modeller yüklendi. Görüntüler okunuyor...")

                    def preprocess_inputs(x):
                        x = np.asarray(x, dtype='float32')
                        x /= 127
                        x -= 1
                        return x

                    pre_img = cv2.imread(pre_path, cv2.IMREAD_COLOR)
                    post_img = cv2.imread(post_path, cv2.IMREAD_COLOR)

                    if pre_img is None:
                        with rasterio.open(pre_path) as src:
                            pre_img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                            if pre_img.dtype != np.uint8:
                                p2, p98 = np.nanpercentile(pre_img[pre_img > 0], [2, 98])
                                pre_img = np.clip((pre_img - p2) / (p98 - p2 + 1e-10) * 255, 0, 255).astype(np.uint8)
                    if post_img is None:
                        with rasterio.open(post_path) as src:
                            post_img = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                            if post_img.dtype != np.uint8:
                                p2, p98 = np.nanpercentile(post_img[post_img > 0], [2, 98])
                                post_img = np.clip((post_img - p2) / (p98 - p2 + 1e-10) * 255, 0, 255).astype(np.uint8)

                    progress.progress(50, text="Bina lokalizasyonu (Loc) çalışıyor...")

                    h, w = pre_img.shape[:2]
                    target_h = ((h - 1) // 256 + 1) * 256
                    target_w = ((w - 1) // 256 + 1) * 256
                    if h != target_h or w != target_w:
                        pre_img_pad = np.zeros((target_h, target_w, 3), dtype=pre_img.dtype)
                        pre_img_pad[:h, :w] = pre_img
                        post_img_pad = np.zeros((target_h, target_w, 3), dtype=post_img.dtype)
                        post_img_pad[:h, :w] = post_img
                        pre_img = pre_img_pad
                        post_img = post_img_pad

                    loc_inp = preprocess_inputs(pre_img.copy())
                    loc_inp_list = [loc_inp, loc_inp[::-1, ...], loc_inp[:, ::-1, ...], loc_inp[::-1, ::-1, ...]]
                    loc_inp_arr = np.asarray(loc_inp_list, dtype='float')
                    loc_tensor = torch.from_numpy(loc_inp_arr.transpose((0, 3, 1, 2))).float().unsqueeze(0)

                    with torch.no_grad():
                        loc_out = loc_wrapper.forward(loc_tensor)
                    loc_out = loc_out.numpy().squeeze()[:h, :w]

                    del loc_wrapper
                    torch.cuda.empty_cache()
                    gc.collect()

                    progress.progress(70, text="Hasar sınıflandırması (Cls) çalışıyor...")

                    cls_inp_img = np.concatenate([pre_img, post_img], axis=2)
                    cls_inp = preprocess_inputs(cls_inp_img)
                    cls_inp_list = [cls_inp, cls_inp[::-1, ...], cls_inp[:, ::-1, ...], cls_inp[::-1, ::-1, ...]]
                    cls_inp_arr = np.asarray(cls_inp_list, dtype='float')
                    cls_tensor = torch.from_numpy(cls_inp_arr.transpose((0, 3, 1, 2))).float().unsqueeze(0)

                    with torch.no_grad():
                        cls_out = cls_wrapper.forward(cls_tensor)
                    cls_out = cls_out.numpy().squeeze()[:h, :w, :]

                    del cls_wrapper
                    torch.cuda.empty_cache()
                    gc.collect()

                    progress.progress(85, text="Sonuçlar işleniyor...")

                    loc_prob = loc_out.astype(np.float32) / 255.0
                    cls_probs = cls_out.astype(np.float32) / 255.0
                    if cls_probs.ndim == 3 and cls_probs.shape[2] >= 4:
                        msk_dmg = cls_probs[:, :, 1:].argmax(axis=2) + 1
                    elif cls_probs.ndim == 3 and cls_probs.shape[2] >= 2:
                        msk_dmg = cls_probs[:, :, 1:].argmax(axis=2) + 1
                    else:
                        msk_dmg = np.ones((h, w), dtype=np.uint8)

                    _thr = [0.38, 0.13, 0.14]
                    msk_loc = (1 * (
                        (loc_prob > _thr[0])
                        | ((loc_prob > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4))
                        | ((loc_prob > _thr[2]) & (msk_dmg > 1))
                    )).astype(np.uint8)

                    from skimage.morphology import square, dilation
                    _msk = msk_dmg == 2
                    if _msk.sum() > 0:
                        _msk = dilation(_msk, square(5))
                        msk_dmg[_msk & msk_dmg == 1] = 2

                    msk_dmg = (msk_dmg * msk_loc).astype(np.uint8)

                    damage_colors = {
                        0: [0, 0, 0, 0],
                        1: [0, 255, 0, 120],
                        2: [255, 255, 0, 150],
                        3: [255, 159, 0, 180],
                        4: [255, 0, 0, 200],
                    }

                    overlay = np.zeros((h, w, 4), dtype=np.uint8)
                    for cls_val, color in damage_colors.items():
                        overlay[msk_dmg == cls_val] = color

                    progress.progress(100, text="Tamamlandı!")

                    unique, counts = np.unique(msk_dmg, return_counts=True)
                    dmg_stats = dict(zip(unique.tolist(), counts.tolist()))
                    total_px = h * w
                    building_px = sum(v for k, v in dmg_stats.items() if k > 0)

                    class_names = {0: "Hasarsız Alan", 1: "Hasarsız Bina", 2: "Hafif Hasar", 3: "Ağır Hasar", 4: "Yıkılmış"}
                    class_emoji = {0: "⬛", 1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Toplam Piksel", f"{total_px:,}")
                    m2.metric("Bina Piksel", f"{building_px:,}")
                    m3.metric("Bina Oranı", f"{building_px / total_px * 100:.1f}%")
                    damaged = dmg_stats.get(2, 0) + dmg_stats.get(3, 0) + dmg_stats.get(4, 0)
                    m4.metric("Hasarlı Bina", f"{damaged:,} px")

                    st.markdown("### Hasar Dağılımı")
                    cols = st.columns(5)
                    for i, cls_val in enumerate([0, 1, 2, 3, 4]):
                        with cols[i]:
                            px = dmg_stats.get(cls_val, 0)
                            pct = px / total_px * 100 if total_px > 0 else 0
                            st.metric(f"{class_emoji[cls_val]} {class_names[cls_val]}", f"{px:,} px ({pct:.1f}%)")

                    st.markdown("### Görüntüler")
                    col1, col2, col3 = st.columns(3)

                    pre_display = pre_img[:h, :w]
                    if pre_display.shape[2] >= 3:
                        pre_display = _norm_rgb(pre_display[:, :, :3])
                    with col1:
                        st.image(pre_display, caption="Pre-Disaster", use_container_width=True)

                    post_display = post_img[:h, :w]
                    if post_display.shape[2] >= 3:
                        post_display = _norm_rgb(post_display[:, :, :3])
                    with col2:
                        st.image(post_display, caption="Post-Disaster", use_container_width=True)

                    from PIL import Image as PILImage
                    pre_pil = PILImage.fromarray(pre_display).convert("RGBA")
                    overlay_pil = PILImage.fromarray(overlay)
                    composite = PILImage.alpha_composite(pre_pil, overlay_pil).convert("RGB")
                    with col3:
                        st.image(np.array(composite), caption="Hasar Overlay", use_container_width=True)

                    st.markdown("### Hasar Haritası (Renk Kodlu)")
                    damage_viz = np.zeros((h, w, 3), dtype=np.uint8)
                    damage_viz[msk_dmg == 0] = [30, 30, 30]
                    damage_viz[msk_dmg == 1] = [0, 200, 0]
                    damage_viz[msk_dmg == 2] = [255, 255, 0]
                    damage_viz[msk_dmg == 3] = [255, 140, 0]
                    damage_viz[msk_dmg == 4] = [255, 0, 0]
                    st.image(damage_viz, caption="Hasar Haritası", use_container_width=True)

                    loc_out_path = str(out / "xview2_loc.png")
                    dmg_out_path = str(out / "xview2_damage.png")
                    overlay_out_path = str(out / "xview2_overlay.png")
                    cv2.imwrite(loc_out_path, msk_loc * 255)
                    cv2.imwrite(dmg_out_path, damage_viz)
                    cv2.imwrite(overlay_out_path, cv2.cvtColor(np.array(composite), cv2.COLOR_RGB2BGR))

                    st.success("Hasar tespiti tamamlandı! Sonuçlar output dizinine kaydedildi.")

                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()


# ── DEMO 06: Değişim Tespiti (AnyChange) ──────────────────────────
elif demo_choice.startswith("06"):
    CHANGE_DIR = Path("/home/openzeka/Desktop/mekansal-veri/pytorch-change-models")
    DEMO_IMG_PRE = CHANGE_DIR / "demo_images" / "t1_img.png"
    DEMO_IMG_POST = CHANGE_DIR / "demo_images" / "t2_img.png"

    st.markdown("""
    **AnyChange ile Değişim Tespiti** — İki uydu görüntüsü arasındaki değişimleri SAM tabanlı AnyChange modeli ile tespit eder.
    """)

    pre_path = None
    post_path = None

    if DEMO_IMG_PRE.exists() and DEMO_IMG_POST.exists():
        pre_path = str(DEMO_IMG_PRE)
        post_path = str(DEMO_IMG_POST)
        st.info(f"Pre: `{DEMO_IMG_PRE.name}` | Post: `{DEMO_IMG_POST.name}`")
    else:
        st.warning("Demo görüntüleri bulunamadı. Lütfen manuel olarak yükleyin.")

    upload_override = st.checkbox("Kendi görüntülerinizi yükleyin")
    if upload_override:
        st.markdown("**Önceki (before) görüntü:**")
        pre_uploaded = st.file_uploader("Before görüntü yükle", type=["tif", "tiff", "png", "jpg"], key="change_before")
        st.markdown("**Sonraki (after) görüntü:**")
        post_uploaded = st.file_uploader("After görüntü yükle", type=["tif", "tiff", "png", "jpg"], key="change_after")
        if pre_uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(pre_uploaded.name).suffix)
            tmp.write(pre_uploaded.read())
            tmp.close()
            pre_path = tmp.name
        if post_uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(post_uploaded.name).suffix)
            tmp.write(post_uploaded.read())
            tmp.close()
            post_path = tmp.name

    if st.button("🚀 Değişim Tespiti Çalıştır", type="primary"):
        if not pre_path or not post_path:
            st.error("Hem before hem after görüntü gerekli!")
        else:
            import gc, torch
            import rasterio
            from skimage.io import imread as skimread
            sys.path.insert(0, str(CHANGE_DIR))
            from torchange.models.segment_any_change import AnyChange

            out = Path(st.session_state["out_dir"])

            with st.spinner("AnyChange ile değişim tespiti çalışıyor (1-3 dk)..."):
                try:
                    def _load_img(p):
                        p = str(p)
                        try:
                            with rasterio.open(p) as src:
                                arr = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                                if arr.dtype != np.uint8:
                                    valid = arr[arr > 0]
                                    if len(valid) > 0:
                                        p2, p98 = np.nanpercentile(valid, [2, 98])
                                        arr = np.clip((arr - p2) / (p98 - p2 + 1e-10) * 255, 0, 255).astype(np.uint8)
                                    else:
                                        arr = np.zeros_like(arr, dtype=np.uint8)
                                return arr
                        except Exception:
                            return skimread(p)

                    img1 = _load_img(pre_path)
                    img2 = _load_img(post_path)

                    max_size = 1024
                    for img_arr in [img1, img2]:
                        if max(img_arr.shape[:2]) > max_size:
                            scale = max_size / max(img_arr.shape[:2])
                            from PIL import Image as _PIL
                            _pil1 = _PIL.fromarray(img1).resize((int(img1.shape[1]*scale), int(img1.shape[0]*scale)), _PIL.BILINEAR)
                            img1 = np.array(_pil1)
                            _pil2 = _PIL.fromarray(img2).resize((int(img2.shape[1]*scale), int(img2.shape[0]*scale)), _PIL.BILINEAR)
                            img2 = np.array(_pil2)
                            break

                    sam_ckpt = os.path.expanduser("~/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth")
                    detector = AnyChange('vit_h', sam_checkpoint=sam_ckpt)
                    detector.make_mask_generator(
                        points_per_side=32,
                        stability_score_thresh=0.95,
                    )
                    detector.set_hyperparameters(
                        change_confidence_threshold=130,
                        use_normalized_feature=True,
                        bitemporal_match=True,
                    )

                    changemasks, _, _ = detector.forward(img1, img2)

                    try:
                        keys = [k for k, _ in changemasks.items()]
                        rles = changemasks['rles'] if 'rles' in keys else []
                    except (TypeError, KeyError):
                        rles = []
                    n_changes = len(rles) if rles is not None else 0

                    change_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
                    if n_changes > 0 and rles:
                        from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask as _rle2mask
                        for rle in rles:
                            try:
                                m = _rle2mask(rle)
                                if m.shape[0] >= change_mask.shape[0] and m.shape[1] >= change_mask.shape[1]:
                                    change_mask[m[:change_mask.shape[0], :change_mask.shape[1]] > 0] = 255
                                else:
                                    change_mask[m > 0] = 255
                            except Exception:
                                pass

                    change_out = str(out / "change_mask.png")
                    import cv2
                    cv2.imwrite(change_out, change_mask)

                    del detector
                    torch.cuda.empty_cache()
                    gc.collect()

                    st.metric("Tespit Edilen Değişim", f"{n_changes} obje")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(img1, caption="Önceki", use_container_width=True)
                    with col2:
                        st.image(img2, caption="Sonraki", use_container_width=True)
                    with col3:
                        if n_changes > 0:
                            st.image(change_mask, caption=f"Değişimler ({n_changes})", use_container_width=True)
                        else:
                            st.info("Değişim bulunamadı.")

                    st.success("Değişim tespiti tamamlandı!")

                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()


# ── DEMO 05: Görünmeyeni Gör ────────────────────────────────────────
elif demo_choice.startswith("05"):
    steps = st.slider("Diffusion Adımları", 10, 200, 100, 10)
    patch_size = st.selectbox("Patch Boyutu", [64, 128, 256], index=1)

    if st.button("🚀 Çalıştır", type="primary"):
        if not input_path:
            st.warning("Önce sidebar'dan veri yükleyin veya konum seçin")
        else:
            with st.spinner("Super Resolution çalışıyor (bu biraz sürebilir)..."):
                import gc, torch
                import geoai
                import rasterio
                from rasterio.plot import show as rioshow

                out = Path(st.session_state["out_dir"])
                with rasterio.open(str(input_path)) as src:
                    h, w = src.height, src.width
                    nb = src.count
                    wh = min(h, 256)
                    ww = min(w, 256)
                    row_off = max(0, (h - wh) // 2)
                    col_off = max(0, (w - ww) // 2)
                    window = (row_off, col_off, wh, ww)

                if nb < 4:
                    with rasterio.open(str(input_path)) as src:
                        profile = src.profile.copy()
                        profile.update(count=4, tiled=True, blockxsize=256, blockysize=256)
                        data = [src.read(b) for b in range(1, nb + 1)]
                        while len(data) < 4:
                            data.append(data[0].copy())
                        tmp_4band = str(out / "input_4band.tif")
                        with rasterio.open(tmp_4band, "w", **profile) as dst:
                            for b_idx, band in enumerate(data, 1):
                                dst.write(band, b_idx)
                    sr_input = tmp_4band
                else:
                    sr_input = str(input_path)

                sr_path = str(out / "super_resolution.tif")
                sr_image, _ = geoai.super_resolution(
                    input_lr_path=sr_input,
                    output_sr_path=sr_path,
                    rgb_nir_bands=[1, 2, 3, 4],
                    window=window,
                    patch_size=patch_size,
                    overlap=16,
                    sampling_steps=steps,
                )

                col1, col2 = st.columns(2)
                with col1:
                    with rasterio.open(str(input_path)) as src:
                        try:
                            from rasterio.windows import Window

                            lr_rgb = src.read(
                                [1, 2, 3],
                                window=Window(
                                    window[1], window[0], window[3], window[2]
                                ),
                            )
                            p2, p98 = np.nanpercentile(lr_rgb[lr_rgb > 0], [2, 98])
                            lr_norm = np.clip((lr_rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
                            st.image(
                                np.transpose(lr_norm, (1, 2, 0)),
                                caption="Orijinal (10m)",
                                use_container_width=True,
                            )
                        except Exception:
                            st.warning("Görüntü okunamadı")
                with col2:
                    with rasterio.open(sr_path) as src:
                        sr_rgb = src.read([1, 2, 3])
                        p2, p98 = np.nanpercentile(sr_rgb[sr_rgb > 0], [2, 98])
                        sr_norm = np.clip((sr_rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
                        st.image(
                            np.transpose(sr_norm, (1, 2, 0)),
                            caption="Super Resolution (2.5m)",
                            use_container_width=True,
                        )

                torch.cuda.empty_cache()
                gc.collect()


# ── DEMO 07: Folium Harita ──────────────────────────────────────────
elif demo_choice.startswith("07"):
    from streamlit_folium import st_folium

    st.markdown("""
    **ArcGIS Public Feature Service + Folium** — Public ArcGIS servislerinde arama yapar, nokta katmanını bulur ve sonucu interaktif haritada gösterir.
    """)

    search_query = st.text_input("Arama sorgusu", value="Chicago crime")
    max_records = st.slider("Maksimum kayıt", 50, 1000, 300, 50)

    @st.cache_data(show_spinner=False, ttl=1800)
    def _search_services(query: str) -> list[dict]:
        payload = _arcgis_get_json(
            "https://www.arcgis.com/sharing/rest/search",
            {
                "q": query,
                "sortField": "numviews",
                "sortOrder": "desc",
                "num": 10,
                "f": "json",
            },
        )
        results = []
        for r in payload.get("results", []):
            if r.get("type") == "Feature Service" and r.get("url"):
                results.append({"title": r.get("title", "?"), "url": r["url"]})
        return results

    if st.button("🔍 Servisleri Ara", key="demo7_search"):
        with st.spinner("ArcGIS'te aranıyor..."):
            try:
                services = _search_services(search_query)
                st.session_state["demo7_services"] = services
                if not services:
                    st.warning("Sonuç bulunamadı")
            except Exception as e:
                st.error(f"Arama hatası: {e}")

    services = st.session_state.get("demo7_services", [])
    if services:
        service_labels = [f"{s['title']}" for s in services]
        selected_idx = st.selectbox(
            "Servis seç",
            range(len(services)),
            format_func=lambda i: service_labels[i],
            key="demo7_service_select",
        )
        selected_service = services[selected_idx]
        st.caption(f"URL: `{selected_service['url']}`")

        if st.button("🚀 Haritayı Oluştur", type="primary", key="demo7_build"):
            with st.spinner("Katman sorgulanıyor..."):
                try:
                    layer_url, layer_name = _find_point_layer_url(selected_service["url"])
                    points = _query_point_features(layer_url, max_records)
                    profile = _build_dataset_profile(points)

                    st.session_state["demo7_points"] = points
                    st.session_state["demo7_profile"] = profile
                    st.session_state["demo7_service_title"] = selected_service["title"]
                    st.session_state["demo7_layer_name"] = layer_name
                    st.session_state["demo7_query"] = search_query

                    st.success(
                        f"Servis: {selected_service['title']} | Katman: {layer_name} | Nokta: {len(points)}"
                    )
                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    points = st.session_state.get("demo7_points")
    profile = st.session_state.get("demo7_profile")
    service_title = st.session_state.get("demo7_service_title")
    layer_name = st.session_state.get("demo7_layer_name")

    if points and profile:
        field_overview = _build_field_overview(points)

        if field_overview:
            with st.expander("Tüm Alanlar ve Örnek Değerler"):
                overview_rows = []
                for f in field_overview:
                    overview_rows.append({
                        "Alan": f["field"],
                        "Tür": f["type"],
                        "Dolu": f["non_null"],
                        "Eşsiz": f["unique"],
                        "Örnek Değerler": " | ".join(str(v)[:40] for v in f["top_values"][:5]),
                    })
                st.dataframe(overview_rows, use_container_width=True, hide_index=True)

        st.subheader("Alan Eşleştirme")
        st.caption("Otomatik algılanan alanları doğruysa onaylayın, yanlışsa düzeltin. Yanlış alan seçimi yanlış sonuç üretir.")
        _all_field_names = sorted({key for _, _, attrs in points for key in attrs.keys()})
        _field_options = ["(yok)"] + _all_field_names
        _inferred = profile["inferred"]

        _col_f1, _col_f2 = st.columns(2)
        with _col_f1:
            _ct_def = _inferred.get("crime_type") or "(yok)"
            _ct_idx = _field_options.index(_ct_def) if _ct_def in _field_options else 0
            _crime_type_sel = st.selectbox("Suç Türü alanı", _field_options, index=_ct_idx, key="demo7_field_crime")

            _desc_def = _inferred.get("description") or "(yok)"
            _desc_idx = _field_options.index(_desc_def) if _desc_def in _field_options else 0
            _desc_sel = st.selectbox("Açıklama alanı", _field_options, index=_desc_idx, key="demo7_field_desc")

            _date_def = _inferred.get("date") or "(yok)"
            _date_idx = _field_options.index(_date_def) if _date_def in _field_options else 0
            _date_sel = st.selectbox("Tarih alanı", _field_options, index=_date_idx, key="demo7_field_date")

        with _col_f2:
            _city_def = _inferred.get("city") or "(yok)"
            _city_idx = _field_options.index(_city_def) if _city_def in _field_options else 0
            _city_sel = st.selectbox("Şehir alanı", _field_options, index=_city_idx, key="demo7_field_city")

            _dist_def = _inferred.get("district") or "(yok)"
            _dist_idx = _field_options.index(_dist_def) if _dist_def in _field_options else 0
            _district_sel = st.selectbox("Bölge/İlçe alanı", _field_options, index=_dist_idx, key="demo7_field_district")

        profile["inferred"]["crime_type"] = _crime_type_sel if _crime_type_sel != "(yok)" else None
        profile["inferred"]["description"] = _desc_sel if _desc_sel != "(yok)" else None
        profile["inferred"]["date"] = _date_sel if _date_sel != "(yok)" else None
        profile["inferred"]["city"] = _city_sel if _city_sel != "(yok)" else None
        profile["inferred"]["district"] = _district_sel if _district_sel != "(yok)" else None
        profile["top_crime_values"] = _top_field_values(points, profile["inferred"]["crime_type"])
        profile["top_city_values"] = _top_field_values(points, profile["inferred"]["city"])
        profile["top_district_values"] = _top_field_values(points, profile["inferred"]["district"])
        st.session_state["demo7_profile"] = profile

        st.subheader("Veri Şeması")
        c1, c2, c3 = st.columns(3)
        c1.metric("Kayıt", f"{profile['record_count']}")
        c2.metric("Servis", service_title or "-")
        c3.metric("Katman", layer_name or "-")

        if profile["top_crime_values"]:
            _crime_counter = Counter()
            _crime_f = profile["inferred"]["crime_type"]
            for _, _, attrs in points:
                v = attrs.get(_crime_f)
                if v is not None and str(v).strip():
                    _crime_counter[_safe_label(v)] += 1
            _top_str = ", ".join(
                f"{v} ({_crime_counter.get(_safe_label(v), 0)})"
                for v in profile["top_crime_values"][:6]
            )
            st.caption(f"En sık suç türleri: {_top_str}")

        current_map_fields = [
            profile["inferred"].get("crime_type"),
            profile["inferred"].get("description"),
            profile["inferred"].get("city"),
            profile["inferred"].get("district"),
            profile["inferred"].get("date"),
        ]
        current_map = _build_folium_map(points, [field for field in current_map_fields if field])

        _map_html_path = str(Path(st.session_state["out_dir"]) / "fullscreen_map.html")
        with open(_map_html_path, "w", encoding="utf-8") as f:
            f.write(current_map.get_root().render())

        _col_preview, _col_btn = st.columns([4, 1])
        with _col_preview:
            st_folium(
                current_map,
                height=800,
                use_container_width=True,
                returned_objects=[],
                key="demo7_current_map",
            )
        with _col_btn:
            st.markdown("### ")
            st.markdown("### ")
            with open(_map_html_path, "r", encoding="utf-8") as f:
                _map_html_content = f.read()
            st.download_button(
                "🖥️ Haritayı HTML\nolarak indir",
                data=_map_html_content,
                file_name="harita_tam_ekran.html",
                mime="text/html",
                key="demo7_download_map",
                help="İndirilen HTML dosyasını tarayıcıda açın — tam ekran, sınırsız zoom, kaydırma",
            )
            if st.button("🖥️ Yeni Sekmede Aç", key="demo7_open_fullscreen", help="Haritayı tam ekran yeni sekmede açar"):
                _port = st.secrets.get("server_port", 8501) if hasattr(st, "secrets") else 8501
                import webbrowser
                import threading
                import http.server
                import socketserver

                class _MapHandler(socketserver.SimpleHTTPRequestHandler):
                    def __init__(self, *a, **kw):
                        super().__init__(*a, directory=str(Path(st.session_state["out_dir"])), **kw)
                    def log_message(self, *a):
                        pass

                if "demo7_http_port" not in st.session_state:
                    _httpd = socketserver.TCPServer(("", 0), _MapHandler)
                    _port = _httpd.server_address[1]
                    st.session_state["demo7_http_port"] = _port
                    threading.Thread(target=_httpd.serve_forever, daemon=True).start()
                else:
                    _port = st.session_state["demo7_http_port"]

                webbrowser.open(f"http://localhost:{_port}/fullscreen_map.html")
                st.success(f"Harita açıldı: http://localhost:{_port}/fullscreen_map.html")

        st.subheader("Harita Agent")
        st.caption("Agent, veri setini doğrudan sorgulamak için araçları kullanır. Harita görselini okumak yerine alttaki veriyi analiz eder.")
        question = st.text_area(
            "Harita hakkında soru sor",
            value="Bu veri setinde en sık görülen suç türleri neler? Hangi bölgelerde yoğunlaşıyor?",
            height=100,
            key="demo7_question",
        )
        show_filtered_map = st.checkbox("Filtrelenmiş sonucu haritada göster", value=True)

        if st.button("💬 Soruyu Cevapla", key="demo7_ask", type="primary"):
            with st.spinner("Agent veriyi sorguluyor..."):
                try:
                    import requests as _req7

                    _pts7 = points
                    _prof7 = profile

                    TOOLS_7 = [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_field_overview",
                                "description": "Veri setindeki tüm alanları, türlerini ve en sık değerleri döner. Önce bunu çağırarak veriyi tanı.",
                                "parameters": {"type": "object", "properties": {}, "required": []},
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "get_crime_distribution",
                                "description": "Suç türlerinin dağılımını (sayı ve yüzde) döner.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "limit": {
                                            "type": "integer",
                                            "description": "Kaç kategori (varsayılan: 10)",
                                        },
                                    },
                                    "required": [],
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "get_location_distribution",
                                "description": "Şehir veya bölge bazında dağılımı döner.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "group_by": {
                                            "type": "string",
                                            "enum": ["city", "district"],
                                            "description": "Gruplama türü",
                                        },
                                        "limit": {
                                            "type": "integer",
                                            "description": "Kaç konum (varsayılan: 10)",
                                        },
                                    },
                                    "required": [],
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "filter_records",
                                "description": "Suç türü veya konuma göre kayıtları filtrele ve istatistik döner.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "crime_keyword": {
                                            "type": "string",
                                            "description": "Suç türü anahtar kelime (örn: theft, assault, burglary)",
                                        },
                                        "location": {
                                            "type": "string",
                                            "description": "Konum adı (örn: Chicago, Austin)",
                                        },
                                    },
                                    "required": [],
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "get_sample_records",
                                "description": "Filtrelenmiş veya tüm kayıtlardan örnek satırlar döner.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "limit": {
                                            "type": "integer",
                                            "description": "Kaç kayıt (varsayılan: 5)",
                                        },
                                        "crime_keyword": {
                                            "type": "string",
                                            "description": "Opsiyonel suç filtresi",
                                        },
                                        "location": {
                                            "type": "string",
                                            "description": "Opsiyonel konum filtresi",
                                        },
                                    },
                                    "required": [],
                                },
                            },
                        },
                    ]

                    def _t7_field_overview():
                        return [
                            {
                                "alan": f["field"],
                                "tur": f["type"],
                                "dolu": f["non_null"],
                                "essiz": f["unique"],
                                "en_sik_degerler": f["top_values"][:5],
                            }
                            for f in field_overview
                        ]

                    def _t7_crime_dist(limit=10):
                        _cf = _prof7["inferred"].get("crime_type")
                        if not _cf:
                            return {"hata": "Suç türü alanı seçili değil. Alan Eşleştirme bölümünden seçin."}
                        _cc = Counter()
                        for _, _, attrs in _pts7:
                            v = attrs.get(_cf)
                            if v is not None and str(v).strip():
                                _cc[_safe_label(v)] += 1
                        _tot = sum(_cc.values())
                        return [
                            {"suç_türü": n, "sayi": c, "yuzde": round(c / _tot * 100, 1)}
                            for n, c in _cc.most_common(limit)
                        ]

                    def _t7_location_dist(group_by="city", limit=10):
                        _lf = _prof7["inferred"].get(group_by)
                        if not _lf:
                            _alt = "district" if group_by == "city" else "city"
                            _lf = _prof7["inferred"].get(_alt)
                        if not _lf:
                            return {"hata": "Konum alanı seçili değil. Alan Eşleştirme bölümünden seçin."}
                        _lc = Counter()
                        for _, _, attrs in _pts7:
                            v = attrs.get(_lf)
                            if v is not None and str(v).strip():
                                _lc[_safe_label(v)] += 1
                        _tot = sum(_lc.values())
                        return [
                            {"konum": n, "sayi": c, "yuzde": round(c / _tot * 100, 1)}
                            for n, c in _lc.most_common(limit)
                        ]

                    def _t7_filter(crime_keyword=None, location=None):
                        _cf = _prof7["inferred"].get("crime_type")
                        _df = _prof7["inferred"].get("description")
                        _cityf = _prof7["inferred"].get("city")
                        _distf = _prof7["inferred"].get("district")
                        _matched = []
                        for lat, lon, attrs in _pts7:
                            if crime_keyword:
                                _hs = " ".join(
                                    _normalize_text(attrs.get(f, ""))
                                    for f in [_cf, _df]
                                    if f
                                )
                                if _normalize_text(crime_keyword) not in _hs:
                                    continue
                            if location:
                                _hs = " ".join(
                                    _normalize_text(attrs.get(f, ""))
                                    for f in [_cityf, _distf]
                                    if f
                                )
                                if _normalize_text(location) not in _hs:
                                    continue
                            _matched.append((lat, lon, attrs))
                        _result = {
                            "eslesen_kayit": len(_matched),
                            "toplam_kayit": len(_pts7),
                            "yuzde": round(len(_matched) / len(_pts7) * 100, 1) if _pts7 else 0,
                        }
                        if _cf:
                            _cc = Counter()
                            for _, _, attrs in _matched:
                                v = attrs.get(_cf)
                                if v is not None and str(v).strip():
                                    _cc[_safe_label(v)] += 1
                            _result["suç_dagilimi"] = [
                                {"suç_türü": n, "sayi": c}
                                for n, c in _cc.most_common(5)
                            ]
                        return _result

                    def _t7_samples(limit=5, crime_keyword=None, location=None):
                        _cf = _prof7["inferred"].get("crime_type")
                        _df = _prof7["inferred"].get("description")
                        _cityf = _prof7["inferred"].get("city")
                        _distf = _prof7["inferred"].get("district")
                        _datef = _prof7["inferred"].get("date")
                        _matched = []
                        for _, _, attrs in _pts7:
                            if crime_keyword:
                                _hs = " ".join(
                                    _normalize_text(attrs.get(f, ""))
                                    for f in [_cf, _df]
                                    if f
                                )
                                if _normalize_text(crime_keyword) not in _hs:
                                    continue
                            if location:
                                _hs = " ".join(
                                    _normalize_text(attrs.get(f, ""))
                                    for f in [_cityf, _distf]
                                    if f
                                )
                                if _normalize_text(location) not in _hs:
                                    continue
                            _matched.append(attrs)
                        _preview = [_cf, _df, _cityf, _distf, _datef]
                        _rows = []
                        for attrs in _matched[:limit]:
                            row = {}
                            for f in _preview:
                                if f:
                                    row[f] = attrs.get(f)
                            _rows.append(row)
                        return {"kayit_sayisi": len(_matched), "ornekler": _rows}

                    _tool_dispatch_7 = {
                        "get_field_overview": lambda a: _t7_field_overview(),
                        "get_crime_distribution": lambda a: _t7_crime_dist(a.get("limit", 10)),
                        "get_location_distribution": lambda a: _t7_location_dist(
                            a.get("group_by", "city"), a.get("limit", 10)
                        ),
                        "filter_records": lambda a: _t7_filter(
                            a.get("crime_keyword"), a.get("location")
                        ),
                        "get_sample_records": lambda a: _t7_samples(
                            a.get("limit", 5), a.get("crime_keyword"), a.get("location")
                        ),
                    }

                    _schema_info_7 = {
                        "alanlar": _prof7["inferred"],
                        "toplam_kayit": len(_pts7),
                        "servis": service_title,
                        "katman": layer_name,
                    }

                    _agent_msgs_7 = [
                        {
                            "role": "system",
                            "content": (
                                "Sen bir suç haritası veri analiz ajansın. Kullanıcının sorusunu cevaplamak için "
                                "araçları kullanarak veriyi doğrudan sorgula. Harita görselini göremezsin, "
                                "ama araçlarla veriye tam erişimin var.\n\n"
                                "İşte çalışma şeklin:\n"
                                "1. Önce get_field_overview ile veri yapısını anla\n"
                                "2. get_crime_distribution ile suç dağılımını al\n"
                                "3. get_location_distribution ile lokasyon dağılımını al\n"
                                "4. Gerekirse filter_records ile detaylı filtreleme yap\n"
                                "5. get_sample_records ile örnek kayıtlar gör\n\n"
                                "Cevabını verirken mutlaka sayı ve yüzde kullan. Veriye dayan. "
                                "Türkçe cevap ver. Kısa ve öz ol.\n\n"
                                "Veri seti özeti:\n"
                                + json.dumps(_schema_info_7, ensure_ascii=False)
                            ),
                        },
                        {"role": "user", "content": question},
                    ]

                    _log_7 = st.empty()
                    _log_lines_7 = []
                    _step_7 = 0
                    _final_answer_7 = ""

                    for _ in range(10):
                        _payload_7 = {
                            "model": LLM_MODEL,
                            "messages": _agent_msgs_7,
                            "tools": TOOLS_7,
                            "tool_choice": "auto",
                            "max_tokens": 2048,
                            "temperature": 0.2,
                        }
                        _resp_7 = _req7.post(LLM_API_URL, json=_payload_7, timeout=120)
                        if _resp_7.status_code != 200:
                            _log_lines_7.append(f"API hatası: {_resp_7.status_code}")
                            break

                        _data_7 = _resp_7.json()
                        _choice_7 = _data_7["choices"][0]
                        _msg_7 = _choice_7["message"]
                        _finish_7 = _choice_7.get("finish_reason", "")

                        if _msg_7.get("content"):
                            _final_answer_7 = _msg_7["content"]

                        if _finish_7 == "stop" or not _msg_7.get("tool_calls"):
                            break

                        _agent_msgs_7.append(_msg_7)

                        for _tc7 in _msg_7["tool_calls"]:
                            _step_7 += 1
                            _fn7 = _tc7["function"]["name"]
                            _fn_args7 = json.loads(_tc7["function"]["arguments"])
                            _log_lines_7.append(
                                f"**Adım {_step_7}:** `{_fn7}({json.dumps(_fn_args7, ensure_ascii=False)[:120]})`"
                            )

                            if _fn7 in _tool_dispatch_7:
                                try:
                                    _tool_out7 = _tool_dispatch_7[_fn7](_fn_args7)
                                    _out_str7 = json.dumps(_tool_out7, ensure_ascii=False, default=str)
                                    _log_lines_7.append(f"  ✅ {_out_str7[:300]}")
                                except Exception as _e7:
                                    _out_str7 = json.dumps({"hata": str(_e7)}, ensure_ascii=False)
                                    _log_lines_7.append(f"  ❌ {_e7}")
                            else:
                                _out_str7 = json.dumps({"hata": f"Bilinmeyen araç: {_fn7}"})

                            _agent_msgs_7.append({
                                "role": "tool",
                                "tool_call_id": _tc7["id"],
                                "content": _out_str7,
                            })

                        _log_7.markdown("\n\n".join(_log_lines_7))

                    _log_7.markdown("\n\n".join(_log_lines_7))

                    if _final_answer_7:
                        st.markdown("### Cevap")
                        st.markdown(_final_answer_7)
                    else:
                        _plan_fb = _plan_question(question, profile)
                        _result_fb = _execute_question_plan(points, profile, _plan_fb)
                        _answer_fb = _answer_question(question, profile, _plan_fb, _result_fb)
                        st.markdown("### Cevap")
                        st.markdown(_answer_fb)

                    if show_filtered_map:
                        _plan_map = _plan_question(question, profile)
                        _result_map = _execute_question_plan(points, profile, _plan_map)
                        if _result_map["matched_points"]:
                            st.markdown("### Filtrelenmiş Harita")
                            _mf = [
                                profile["inferred"].get("crime_type"),
                                profile["inferred"].get("description"),
                                profile["inferred"].get("city"),
                                profile["inferred"].get("district"),
                                profile["inferred"].get("date"),
                            ]
                            _fmap = _build_folium_map(
                                _result_map["matched_points"],
                                [f for f in _mf if f],
                            )
                            _fmap_html_path = str(Path(st.session_state["out_dir"]) / "filtered_map.html")
                            with open(_fmap_html_path, "w", encoding="utf-8") as f:
                                f.write(_fmap.get_root().render())
                            _fc1, _fc2 = st.columns([4, 1])
                            with _fc1:
                                st_folium(
                                    _fmap,
                                    height=800,
                                    use_container_width=True,
                                    returned_objects=[],
                                    key="demo7_filtered_map",
                                )
                            with _fc2:
                                st.markdown("### ")
                                st.markdown("### ")
                                with open(_fmap_html_path, "r", encoding="utf-8") as f:
                                    _fmap_html_content = f.read()
                                st.download_button(
                                    "🖥️ Filtrelenmiş\nHaritayı İndir",
                                    data=_fmap_html_content,
                                    file_name="filtrelenmis_harita.html",
                                    mime="text/html",
                                    key="demo7_download_filtered_map",
                                    help="İndirilen HTML dosyasını tarayıcıda açın",
                                )

                    st.success(f"Agent tamamlandı ({_step_7} adım)")

                except Exception as e:
                    st.error(f"Hata: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ── FOOTER ───────────────────────────────────────────────────────────
st.divider()
with st.expander("Desteklenen Formatlar"):
    st.markdown("""
    **Raster:** GeoTIFF (.tif/.tiff), COG, ERDAS (.img), ECW, MrSID, JPEG, PNG, WebP, BMP
    **Vektör:** Shapefile (.shp), GeoJSON, GeoPackage (.gpkg), KML/KMZ, GPX, WKT
    **Uydu:** Sentinel-2, Landsat, NAIP, Planet (STAC üzerinden)
    """)
with st.expander("Kullanılabilir Örnek Veriler"):
    for name, url in SAMPLE_DATA.items():
        st.markdown(f"- **{name}**: `{url.split('/')[-1]}`")
with st.expander("Konumlar"):
    for name, data in LOCATIONS.items():
        st.markdown(f"- **{name}**: {data['center']}")
