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
from pathlib import Path

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


DEMOS = {
    "01 — Metinle Segmentasyon": "CLIPSeg / GroundedSAM / SamGeo / xView2-Loc",
    "02 — Görüntü Analizi": "VLM ile Uydu Görüntüsü Analizi",
    "03 — Ajan Analizi": "AI Ajan ile Uçtan Uca Analiz",
    "04 — Hasar Tespiti": "xView2 Bina Hasar Sınıflandırma",
    "05 — Süper Çözünürlük": "4x Uydu Görüntüsü Süper Çözünürlük",
    "06 — Değişim Tespiti": "AnyChange ile Değişim Tespiti",
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

    if loc_method == "Şehir Seç":
        city = st.selectbox("Şehir", list(LOCATIONS.keys()))
        loc_data = LOCATIONS[city]
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[loc_data["center"][1], loc_data["center"][0]], zoom_start=12
        )
        folium.Marker(
            [loc_data["center"][1], loc_data["center"][0]],
            popup=city,
            icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        ).add_to(m)
        st_folium(m, height=300, use_container_width=True, returned_objects=[])
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

        m = folium.Map(location=[lat, lon], zoom_start=12)
        folium.Marker(
            [lat, lon],
            popup=f"{lat:.4f}°N, {lon:.4f}°E",
            icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
        ).add_to(m)
        st_folium(m, height=300, use_container_width=True, returned_objects=[])
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
        m = folium.Map(location=map_center, zoom_start=5 if sel_lat is None else 13)

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
    threshold = st.slider("GroundedSAM Eşiği", 0.05, 0.5, 0.2, 0.05)

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
                        threshold=threshold,
                    )
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
    default_qs = [
        "Describe this satellite image in detail.",
        "How many buildings can you see?",
        "Is there any water body?",
        "What type of vegetation is present?",
        "What is the main land use?",
    ]
    questions_text = st.text_area(
        "Sorular (her satıra bir tane)", value="\n".join(default_qs), height=150
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
                        "model": "GLM-5.1-FP8",
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
                                    {"type": "text", "text": q},
                                ],
                            }
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.3,
                    }
                    resp = req.post(
                        "http://192.168.1.200:8000/v1/chat/completions",
                        json=payload,
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["choices"][0]["message"]["content"]
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

        API_URL = "http://192.168.1.200:8000/v1/chat/completions"
        MODEL = "GLM-5.1-FP8"
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
    TEST_PRE = XVIEW2_DIR / "tests" / "data" / "input" / "pre"
    TEST_POST = XVIEW2_DIR / "tests" / "data" / "input" / "post"

    st.markdown("""
    **xView2 Bina Hasar Tespiti** — Afet sonrası uydu görüntülerinden bina hasarını sınıflandırır.
    
    Hasar sınıfları: 🟢 Hasarsız · 🟡 Hafif Hasar · 🟠 Ağır Hasar · 🔴 Yıkılmış
    """)

    use_test_data = st.checkbox("Örnek veriyi kullan (xView2 test verisi)", value=True)

    pre_path = None
    post_path = None

    if use_test_data:
        pre_files = sorted(TEST_PRE.glob("*.tif")) if TEST_PRE.exists() else []
        post_files = sorted(TEST_POST.glob("*.tif")) if TEST_POST.exists() else []
        if pre_files and post_files:
            pre_path = str(pre_files[0])
            post_path = str(post_files[0])
            st.info(f"Pre: `{Path(pre_path).name}` | Post: `{Path(post_path).name}`")
            if len(pre_files) > 1:
                sel = st.selectbox("Pre görüntü seç", range(len(pre_files)), format_func=lambda i: pre_files[i].name)
                pre_path = str(pre_files[sel])
            if len(post_files) > 1:
                sel = st.selectbox("Post görüntü seç", range(len(post_files)), format_func=lambda i: post_files[i].name)
                post_path = str(post_files[sel])
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
    XVIEW2_DIR = Path("/home/openzeka/Desktop/mekansal-veri/xView2-deploy")
    DEMO_IMG_PRE = CHANGE_DIR / "demo_images" / "t1_img.png"
    DEMO_IMG_POST = CHANGE_DIR / "demo_images" / "t2_img.png"
    XVIEW2_PRE = XVIEW2_DIR / "tests" / "data" / "input" / "pre"
    XVIEW2_POST = XVIEW2_DIR / "tests" / "data" / "input" / "post"

    st.markdown("""
    **AnyChange ile Değişim Tespiti** — İki uydu görüntüsü arasındaki değişimleri SAM tabanlı AnyChange modeli ile tespit eder.
    
    Veri kaynağı olarak pytorch-change-models repo demosu veya xView2-deploy test verisi kullanılabilir.
    """)

    data_src = st.radio("Veri Kaynağı", ["pytorch-change-models demo görüntüleri", "xView2-deploy test verisi"], horizontal=True)

    pre_path = None
    post_path = None

    if data_src.startswith("pytorch"):
        if DEMO_IMG_PRE.exists() and DEMO_IMG_POST.exists():
            pre_path = str(DEMO_IMG_PRE)
            post_path = str(DEMO_IMG_POST)
            st.info(f"Pre: `{DEMO_IMG_PRE.name}` | Post: `{DEMO_IMG_POST.name}`")
        else:
            st.warning("Demo görüntüleri bulunamadı. Lütfen manuel olarak yükleyin.")
    else:
        pre_files = sorted(XVIEW2_PRE.glob("*.tif")) if XVIEW2_PRE.exists() else []
        post_files = sorted(XVIEW2_POST.glob("*.tif")) if XVIEW2_POST.exists() else []
        if pre_files and post_files:
            sel = st.selectbox("Pre görüntü seç", range(len(pre_files)), format_func=lambda i: pre_files[i].name)
            pre_path = str(pre_files[sel])
            sel2 = st.selectbox("Post görüntü seç", range(len(post_files)), format_func=lambda i: post_files[i].name)
            post_path = str(post_files[sel2])
            st.info(f"Pre: `{Path(pre_path).name}` | Post: `{Path(post_path).name}`")
        else:
            st.warning("xView2 test verisi bulunamadı. Lütfen manuel olarak yükleyin.")

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


# ── DEMO 07: Görünmeyeni Gör ────────────────────────────────────────
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
