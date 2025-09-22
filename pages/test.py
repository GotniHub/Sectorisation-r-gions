# test_algo.py
# ------------------------------------------------------------
# Page de test : sectorisation multi-critères (Géo + CA + Visites + Nb Magasins)
# • Charge un Excel avec au moins : "Code du client", "Departement", "Nb Visite", "CA 2024"
# • Charge un GeoJSON départements (propriété "code" et "nom")
# • Calcule centroïdes en EPSG:2154 (Lambert-93), reprojette en WGS84, clusterise, affiche carte + tableau
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import copy
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide", page_title="TEST — Sectorisation multi-critères")

st.title("🧪 Test sectorisation multi-critères (départements)")

# -----------------------------
# Sidebar : chargements & params
# -----------------------------
st.sidebar.header("📂 Données")
uploaded_file = st.sidebar.file_uploader("Excel magasins", type=["xlsx", "xls"])
geojson_path = st.sidebar.text_input("Chemin GeoJSON", value="geoson.geojson")

st.sidebar.header("⚙️ Paramètres")
nb_clusters = st.sidebar.slider("Nombre de zones", min_value=2, max_value=10, value=5, step=1)
diviseur_etp = st.sidebar.number_input("Diviseur ETP (visites / ETP)", value=949, min_value=1)

st.sidebar.header("⚖️ Pondérations des critères")
w_geo     = st.sidebar.slider("Poids Géo (lat/lon)", 0.0, 5.0, 1.0, 0.1)
w_ca      = st.sidebar.slider("Poids CA 2024",        0.0, 5.0, 1.0, 0.1)
w_visites = st.sidebar.slider("Poids Nb Visite",      0.0, 5.0, 1.0, 0.1)
w_clients = st.sidebar.slider("Poids Nb Magasins",    0.0, 5.0, 1.0, 0.1)

st.sidebar.caption("Astuce : commence par w_geo=1.5, w_ca=0.5, w_visites=1, w_clients=0.5")
# ---- Priorités (tie-break quand les poids sont égaux) + Debug ----
st.sidebar.markdown("### 🥇 Priorité des critères")
options = ["Géo", "CA 2024", "Nb Visite", "Nb Magasins"]

p1 = st.sidebar.selectbox("Priorité 1 (plus forte)", options, index=0, key="prio1")
opts2 = [o for o in options if o != p1]
p2 = st.sidebar.selectbox("Priorité 2", opts2, index=0, key="prio2")
opts3 = [o for o in opts2 if o != p2]
p3 = st.sidebar.selectbox("Priorité 3", opts3, index=0, key="prio3")
opts4 = [o for o in opts3 if o != p3]
p4 = st.sidebar.selectbox("Priorité 4 (plus faible)", opts4, index=0, key="prio4")

priority_order = [p1, p2, p3, p4]

# Épsilon décroissant (très petit) : 1e-3 > 1e-6 > 1e-9 > 1e-12
eps_vals = [1e-3, 1e-6, 1e-9, 1e-12]
eps = {crit: eps_vals[i] for i, crit in enumerate(priority_order)}

# Poids de base (issus des sliders)
w_map = {"Géo": w_geo, "CA 2024": w_ca, "Nb Visite": w_visites, "Nb Magasins": w_clients}

# On n'ajoute l'epsilon que si le poids > 0 (si tu mets 0, le critère reste ignoré)
def with_eps(crit):
    return w_map[crit] + (eps[crit] if w_map[crit] > 0 else 0.0)

w_final = {crit: with_eps(crit) for crit in w_map}

# (optionnel) mode "priorité stricte (lexicographique)" très marqué
strict = st.sidebar.toggle("Priorité stricte (lexicographique)", value=False,
                           help="Multiplie par une grande échelle selon le rang. À utiliser si tu veux un ordre très tranché.")
if strict:
    # rang élevé => facteur beaucoup plus grand
    scale = {priority_order[0]: 1e6, priority_order[1]: 1e4, priority_order[2]: 1e2, priority_order[3]: 1}
    w_final = {k: (w_map[k] * (scale[k] if w_map[k] > 0 else 0.0)) for k in w_map}

# Applique ces poids (w_final) plus loin dans ton code :
# X_std["lat"]          *= w_final["Géo"]
# X_std["lon"]          *= w_final["Géo"]
# X_std["CA 2024"]      *= w_final["CA 2024"]
# X_std["Nb Visite"]    *= w_final["Nb Visite"]
# X_std["Nb Magasins"]  *= w_final["Nb Magasins"]

# ---------- Panneau DEBUG : voir l'effet concret ----------
import pandas as pd
debug_rows = []
for i, crit in enumerate(options, start=1):
    base = w_map[crit]
    added = (w_final[crit] - base) if not strict else (w_final[crit] - base)  # info indicative
    # Influence ~ poids^2 (Ward). Pour la Géo, lat+lon => ~2 * poids^2
    infl = (2 * (w_final[crit]**2)) if crit == "Géo" else (w_final[crit]**2)
    debug_rows.append({
        "Rang de priorité": priority_order.index(crit) + 1,
        "Critère": crit,
        "Poids base": round(base, 6),
        "Epsilon ajouté": (round(added, 12) if not strict else "— (mode strict)"),
        "Poids final": round(w_final[crit], 6) if not strict else f"{w_final[crit]:.4g}",
        "Influence approx (σ²)": round(infl, 4),
    })

st.sidebar.markdown("#### 🔎 Détails pondérations & priorités")
st.sidebar.dataframe(pd.DataFrame(debug_rows).sort_values("Rang de priorité"),
                     use_container_width=True)


# -----------------------------
# Helpers
# -----------------------------
def format_dep(val: object) -> str | None:
    """Normalise le code département en chaîne : 01..95, garde 3 chiffres (971..), gère floats."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if s in {"2A", "2B"}:
        # Dans ce test on regroupe Corse en '20' pour rester cohérent avec le GeoJSON simplifié
        s = "20"
    # gérer floats style '59.0'
    try:
        f = float(s)
        i = int(round(f))
        if i >= 100:     # DOM
            return str(i)
        return str(i).zfill(2)
    except:
        # si c'est déjà '20' / '75' etc.
        if s.isdigit():
            if len(s) == 1:
                return s.zfill(2)
            return s
        if s == "20":
            return s
        return s  # fallback

def build_zone_table(zones_dict: dict, df_base: pd.DataFrame, diviseur: float) -> pd.DataFrame:
    rows = []
    dfb = df_base.copy()
    dfb["Departement"] = dfb["Departement"].astype(str)
    # éviter NaN et assurer zfill(2) pour 2 chiffres, ne pas toucher DOM
    dfb["Departement"] = dfb["Departement"].apply(lambda x: x if len(x) == 3 else x.zfill(2))

    if "Code du client" in dfb.columns:
        dfb["Code du client"] = dfb["Code du client"].astype(str)

    for zone, deps in zones_dict.items():
        deps_norm = []
        for d in deps:
            d = str(d)
            deps_norm.append(d if len(d) == 3 else d.zfill(2))

        zdf = dfb[dfb["Departement"].isin(deps_norm)].copy()
        if "Code du client" in zdf.columns:
            zdf = zdf.drop_duplicates(subset=["Code du client"])

        ca = float(zdf.get("CA 2024", pd.Series(dtype=float)).sum())
        visites = float(zdf.get("Nb Visite", pd.Series(dtype=float)).sum())
        nb_clients = int(zdf["Code du client"].nunique()) if "Code du client" in zdf.columns else int(zdf.shape[0])
        etp = round(visites / diviseur, 2) if diviseur > 0 else 0.0

        rows.append({
            "Zone": zone,
            "Départements": ", ".join(sorted(deps_norm)),
            "Nb Clients": nb_clients,
            "CA 2024 (€)": ca,
            "Nb Visites": visites,
            "ETP estimé": etp,
            "Nb Visites / ETP": round(visites / etp, 2) if etp > 0 else 0.0,
            "Nb Clients / ETP": round(nb_clients / etp, 2) if etp > 0 else 0.0,
        })

    out = pd.DataFrame(rows).sort_values("Zone")
    return out[["Zone","Départements","Nb Clients","CA 2024 (€)","Nb Visites","ETP estimé","Nb Visites / ETP","Nb Clients / ETP"]]

# -----------------------------
# Charger données
# -----------------------------
if uploaded_file is None:
    st.info("➡️ Charge un **Excel** pour continuer.")
    st.stop()

# Excel
df = pd.read_excel(uploaded_file)
# Normalise colonnes & valeurs utiles
df.columns = (
    pd.Series(df.columns).astype(str)
    .str.replace("\u00A0", " ", regex=False)
    .str.strip()
)
# Détecte colonne département
DEP_COL_CANDIDATES = ["Departement", "Département", "Dépt", "Dept", "Département Client"]
DEP_COL = next((c for c in DEP_COL_CANDIDATES if c in df.columns), None)
if DEP_COL is None:
    st.error("Colonne département introuvable (essayé : Departement / Département / Dépt / Dept / Département Client).")
    st.stop()

# Build base df_f
df_f = df.copy()
df_f["Departement"] = df_f[DEP_COL].apply(format_dep)
df_f["Nb Visite"] = pd.to_numeric(df_f.get("Nb Visite", 0), errors="coerce").fillna(0)
df_f["CA 2024"] = pd.to_numeric(df_f.get("CA 2024", 0), errors="coerce").fillna(0)
df_f["Nb Magasins"] = 1
if "Code du client" in df_f.columns:
    df_f["Code du client"] = df_f["Code du client"].astype(str)

# GeoJSON
if not Path(geojson_path).exists():
    st.error(f"GeoJSON introuvable : {geojson_path}")
    st.stop()

with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# Regrouper 2A/2B en "20" pour correspondre à notre normalisation
for feature in geojson_data["features"]:
    code = str(feature["properties"]["code"]).upper().strip()
    if code in ["2A", "2B"]:
        feature["properties"]["code"] = "20"

gdf_dept = gpd.GeoDataFrame.from_features(geojson_data["features"]).set_crs("EPSG:4326")

# -----------------------------
# Centroïdes en Lambert-93 -> WGS84 (corrige le warning CRS)
# -----------------------------
gdf_dept["Departement"] = (
    gdf_dept["code"].astype(str).str.strip().str.upper()
           .replace({"2A": "20", "2B": "20"})
)

gdf_diss = gdf_dept.dissolve(by="Departement", as_index=True)
gdf_diss_m = gdf_diss.to_crs(epsg=2154)              # métrique
centroids_m = gdf_diss_m.geometry.centroid
centroids_geo = gpd.GeoSeries(centroids_m, crs="EPSG:2154").to_crs(epsg=4326)

centroids_df = pd.DataFrame({
    "Departement": gdf_diss_m.index.astype(str),
    "lon": centroids_geo.x.values,
    "lat": centroids_geo.y.values
}).reset_index(drop=True)

# -----------------------------
# Agrégation par département (après éventuelles exclusions ; ici on prend tout df_f)
# -----------------------------
dept_data = df_f.groupby("Departement", dropna=True).agg({
    "Nb Magasins": "sum",
    "Nb Visite": "sum",
    "CA 2024": "sum"
}).reset_index()

merged = pd.merge(dept_data, centroids_df, on="Departement", how="left").dropna(subset=["lat", "lon"])
if merged.empty:
    st.error("Aucun département n'a pu être mergé avec le GeoJSON. Vérifie les codes départements.")
    st.stop()

# -----------------------------
# Clustering multi-critères (Ward)
# -----------------------------
feats_cols = ["lat", "lon", "CA 2024", "Nb Visite", "Nb Magasins"]
X_raw = merged[feats_cols].fillna(0)

X_std = pd.DataFrame(
    StandardScaler().fit_transform(X_raw),
    columns=feats_cols,
    index=merged.index
)
X_std["lat"]          *= w_final["Géo"]
X_std["lon"]          *= w_final["Géo"]
X_std["CA 2024"]      *= w_final["CA 2024"]
X_std["Nb Visite"]    *= w_final["Nb Visite"]
X_std["Nb Magasins"]  *= w_final["Nb Magasins"]


agglo = AgglomerativeClustering(n_clusters=nb_clusters, linkage="ward")
labels = agglo.fit_predict(X_std.to_numpy())

uniq = sorted(np.unique(labels))
remap = {old: i for i, old in enumerate(uniq)}
merged["Cluster"] = [remap[l] for l in labels]
merged["Zone"] = merged["Cluster"].apply(lambda x: f"Zone {chr(65 + x)}")

# Mapping Zone -> liste de départements (zfill pour 2 chiffres ; DOM inchangé)
def _zfill_dep(s: str) -> str:
    s = str(s)
    return s if len(s) == 3 else s.zfill(2)

zones_dict = (
    merged.assign(Departement=merged["Departement"].apply(_zfill_dep))
          .groupby("Zone")["Departement"]
          .apply(list).to_dict()
)
# ========== DIAGNOSTIC COHÉRENCE (robuste) ==========
import numpy as np
import pandas as pd
from shapely.ops import unary_union

# --- Seuils réglables dans la sidebar
MIN_POLSBY = st.sidebar.slider("Seuil compacité Polsby", 0.05, 0.40, 0.20, 0.01,    
            help="Indice 0–1 : 1 = zone très compacte (ronde), 0 = forme dentelée/allongée. "
            "Mesure si la zone est ronde ou tordue. Montez pour exiger des zones plus rondes."
                               )
MIN_HULL_RATIO = st.sidebar.slider("Seuil aire/convex hull", 0.30, 0.90, 0.55, 0.01,
                                help="Aire(zone) / Aire(enveloppe convexe). Proche de 1 = peu de trous/concavités. "
                                "Mesure si la zone remplit bien son enveloppe (pas de trous/croissants). Montez pour tolérer moins de trous.")
MAX_SPAN_KM = st.sidebar.slider("Seuil étalement max (km)", 200, 900, 550, 10,
                                help="Distance maximale entre deux départements de la zone. "
                                "Limite les zones trop longues. Baissez le seuil pour être plus strict.")
MIN_DOMINANT_RATIO = st.sidebar.slider("Seuil composante dominante", 0.50, 1.00, 0.80, 0.01,
                                        help="Part minimale de la zone qui doit être d’un seul tenant (ex: 0.80 = au moins 80% collé).")

smooth_on = st.sidebar.toggle(
    "Lisser les contours pour Polsby", value=True,
    help="Buffer/débuffer: Adoucit les côtes avant la mesure de compacité pour éviter de fausses alertes."
)
smooth_buffer_m = st.sidebar.slider("Rayon de lissage (m)", 0, 5000, 2000, 100)

# Mode d’alerte
strict_mode = st.sidebar.toggle(
    "Mode strict (alerte si un seul critère est KO)", value=False,
    help="OFF : alerte si Polsby bas + (span ou hull KO). ON : alerte si n'importe quel critère est KO."
)
# ⬇️ à placer dans la SIDEBAR, après MIN_* / MAX_* + smooth_on + strict_mode
allow_islands = st.sidebar.toggle(
    "Tolérer les petits îlots si composante dominante ≥ seuil",
    value=True,
    help="Si activé : pas d’alerte quand il reste un petit îlot, tant que la composante principale dépasse le seuil."
)

def _zfmt(s: object) -> str:
    s = str(s).strip().upper()
    return s if len(s) == 3 else s.zfill(2)  # DOM en 3 chiffres, sinon 2

# Zonage courant
zones_src = st.session_state.zones_modifiables if "zones_modifiables" in st.session_state and st.session_state.zones_modifiables else zones_dict
zones = {z: sorted({_zfmt(d) for d in deps}) for z, deps in zones_src.items()}

# GeoDataFrame dissous indexé sur codes zfill
gdf_idx = gdf_diss_m.copy()
gdf_idx.index = gdf_idx.index.astype(str).map(_zfmt)

# Graphe d’adjacence par ARÊTE partagée
tol_edge_m = 1.0
all_idx = list(gdf_idx.index)
adj = {i: set() for i in all_idx}
for a in range(len(all_idx)):
    i = all_idx[a]; gi = gdf_idx.geometry.loc[i]
    for b in range(a+1, len(all_idx)):
        j = all_idx[b]; gj = gdf_idx.geometry.loc[j]
        shared = gi.boundary.intersection(gj.boundary)
        if getattr(shared, "length", 0.0) > tol_edge_m:
            adj[i].add(j); adj[j].add(i)

def _components(nodes, adj_map):
    nodes = set(nodes)
    seen, comps = set(), []
    for s in nodes:
        if s in seen:
            continue
        stack = [s]; comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u)
            for v in adj_map.get(u, ()):
                if v in nodes and v not in seen:
                    stack.append(v)
        comps.append(sorted(comp))
    return comps

def _zone_metrics(zone_deps):
    keep = [d for d in zone_deps if d in gdf_idx.index]
    if not keep:
        return [], 1.0, True, 0.0, 0.0, 0.0  # vide => flag comme mauvais

    # composantes
    comps = _components(keep, adj)
    n = sum(len(c) for c in comps)
    largest = max((len(c) for c in comps), default=0)
    dominant_ratio = (largest / n) if n else 1.0
    fragmented = (len(comps) > 1) or (dominant_ratio < MIN_DOMINANT_RATIO)

    # géométrie unie
    g_zone = gdf_idx.loc[keep].geometry
    union = unary_union(list(g_zone))
    area = union.area

    # périmètre lissé (optionnel)
    if smooth_on and smooth_buffer_m > 0:
        smoothed = union.buffer(smooth_buffer_m).buffer(-smooth_buffer_m)
        geom_for_perim = smoothed if smoothed.is_valid and not smoothed.is_empty else union
    else:
        geom_for_perim = union

    perim = geom_for_perim.length
    polsby = (4*np.pi*area/(perim**2)) if perim > 0 else 0.0

    hull = union.convex_hull
    hull_ratio = (area/hull.area) if hull.area > 0 else 1.0

    # étalement max (km) via centroïdes 2154
    cent = g_zone.centroid
    xs, ys = cent.x.values, cent.y.values
    span_m = 0.0
    for i in range(len(xs)):
        dx = xs[i:] - xs[i]
        dy = ys[i:] - ys[i]
        if len(dx):
            d = np.sqrt(dx*dx + dy*dy).max()
            span_m = float(max(span_m, d))
    span_km = span_m / 1000.0

    return comps, dominant_ratio, fragmented, polsby, hull_ratio, span_km

bad_any = False
debug_rows = []

for z, deps_zone in zones.items():
    # métriques de base (tu as déjà _zone_metrics)
    comps, dominant_ratio, _fragmented_unused, polsby, hull_ratio, span_km = _zone_metrics(deps_zone)

    # ----- Fragmentation / contiguïté (avec tolérance d'îlots) -----
    sizes = [len(c) for c in comps] if comps else []
    n = sum(sizes) if sizes else 0
    largest = max(sizes) if sizes else 0
    multi_comp = (len(comps) > 1)
    ratio_bad = (dominant_ratio < MIN_DOMINANT_RATIO)

    # tolérance des îlots : si activée, on ne badgera que si le ratio est insuffisant
    if allow_islands:
        fragmented = ratio_bad
    else:
        fragmented = multi_comp or ratio_bad

    # message clair (corrige le signe et explique la cause)
    frag_msg = ""
    if fragmented:
        sign = "<" if ratio_bad else "≥"
        frag_msg = (
            f"fragmentée ({len(comps)} composantes; plus grande={largest}/{n}, "
            f"ratio={dominant_ratio:.2f} {sign} {MIN_DOMINANT_RATIO:.2f}"
            + (", îlot toléré" if (multi_comp and not ratio_bad and allow_islands) else "")
            + (", îlot(s) non toléré(s)" if (multi_comp and not ratio_bad and not allow_islands) else "")
            + ")"
        )

    # ----- Autres critères de forme -----
    pols_bad = (polsby < MIN_POLSBY)
    span_bad = (span_km > MAX_SPAN_KM)
    hull_bad = (hull_ratio < MIN_HULL_RATIO)

    # Mode strict : alerte si un seul critère est KO
    if strict_mode:
        zone_bad = fragmented or pols_bad or span_bad or hull_bad
    else:
        # Mode normal : Polsby bas + (span OU hull KO) OU fragmentation
        zone_bad = fragmented or (pols_bad and (span_bad or hull_bad))

    # ----- Rendu par zone -----
    if zone_bad:
        bad_any = True
        msgs = []
        if frag_msg:
            msgs.append(frag_msg)
        if pols_bad or span_bad or hull_bad:
            sub = [f"Polsby={polsby:.2f}"]
            if span_bad: sub.append(f"span={span_km:.0f}km>{MAX_SPAN_KM}")
            if hull_bad: sub.append(f"hull={hull_ratio:.2f}<{MIN_HULL_RATIO}")
            msgs.append(", ".join(sub))
        st.warning(f"⚠️ Zone **{z}** : " + " · ".join(msgs))

    # ----- Tableau debug -----
    debug_rows.append({
        "Zone": z,
        "#deps": len(deps_zone),
        "#composantes": len(comps),
        "dom_ratio": round(dominant_ratio, 3),
        "Polsby": round(polsby, 3),
        "Hull ratio": round(hull_ratio, 3),
        "Span (km)": round(span_km, 1),
        "frag_bad": fragmented,
        "pols_bad": pols_bad,
        "span_bad": span_bad,
        "hull_bad": hull_bad,
        "zone_bad": zone_bad,
    })



# ======= Détails en MODALE si alertes, sinon POPOVER =======
import pandas as pd
dbg = pd.DataFrame(debug_rows).sort_values("Zone")

# Construire les messages d'alerte par zone depuis le tableau debug
zone_messages = []
for r in dbg.to_dict("records"):
    msgs = []
    if r.get("frag_bad"): msgs.append("fragmentation")
    shape_bits = []
    if r.get("pols_bad"):  shape_bits.append(f"Polsby={r['Polsby']:.2f}")
    if r.get("span_bad"):  shape_bits.append(f"span={r['Span (km)']:.0f} km")
    if r.get("hull_bad"):  shape_bits.append(f"hull={r['Hull ratio']:.2f}")
    if shape_bits: msgs.append(", ".join(shape_bits))
    if msgs: zone_messages.append(f"⚠️ {r['Zone']} : " + " · ".join(msgs))

# Compatibilité : st.dialog (stable) ou experimental_dialog (anciennes versions)
_open_dialog = getattr(st, "dialog", None) or st.experimental_dialog

@_open_dialog("⚠️ Des incohérences ont été détectées")
def show_alerts_dialog(messages: list[str], df: pd.DataFrame):
    if messages:
        st.markdown("### Alertes (voir détails par zone ci-dessus et tableau debug) :")
        for m in messages:
            st.warning(m)
    else:
        st.info("Aucune alerte détectée selon les seuils courants.")
    st.markdown("### Tableau debug — métriques par zone")
    st.dataframe(df, use_container_width=True)

# Auto-ouverture de la modale UNIQUEMENT s'il y a des alertes
if zone_messages:
    # (optionnel) éviter d'ouvrir à chaque rerun — on l’ouvre 1 fois automatiquement
    # Auto-ouverture de la modale À CHAQUE RERUN s'il y a des alertes
    if zone_messages:
        show_alerts_dialog(zone_messages, dbg)  # s’ouvre systématiquement
    # Bouton pour rouvrir la modale à la demande
    if st.button(f"🔎 Détails — {len(zone_messages)} alerte(s)"):
        show_alerts_dialog(zone_messages, dbg)
else:
    # Tout est OK : popover discret pour consulter les détails si besoin
    with st.popover("ℹ️ Tout est OK (Voir détails)"):
        st.success("✅ Cohérence spatiale : contiguïté OK et zones suffisamment compactes selon les seuils.")
        st.info("Aucune alerte détectée selon les seuils courants.")
        st.markdown("### Tableau debug — métriques par zone")
        st.dataframe(dbg, use_container_width=True)


# -----------------------------
# Affichage : Carte + Tableau
# -----------------------------
st.subheader("🗺️ Carte — Clustering multi-critères")

palette = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
           "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
           "#008080", "#e6beff"]
zone_color_map = {f"Zone {chr(65+i)}": palette[i % len(palette)] for i in range(nb_clusters)}

geojson_multi = copy.deepcopy(geojson_data)
for feature in geojson_multi["features"]:
    code_dep = str(feature["properties"]["code"]).strip().upper()
    code_dep = code_dep if len(code_dep) == 3 else code_dep.zfill(2)
    zone_name = next((z for z, deps in zones_dict.items() if code_dep in deps), None)
    feature["properties"]["Zone_multi"] = zone_name or "Non assigné"

def style_multi(feature):
    z = feature["properties"].get("Zone_multi")
    return {
        'fillColor': zone_color_map.get(z, "#d9d9d9"),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.6
    }

m_multi = folium.Map(location=[46.7, 2.5], zoom_start=6)
folium.GeoJson(
    geojson_multi,
    name="Départements",
    style_function=style_multi,
    tooltip=folium.GeoJsonTooltip(fields=["nom", "code", "Zone_multi"], aliases=["Nom", "Code", "Zone"])
).add_to(m_multi)

st_folium(m_multi, width=1100, height=620)

st.subheader("📊 Indicateurs par zone — multi-critères")
zone_df_multi = build_zone_table(zones_dict, df_f, diviseur_etp)
def fmt_fr_int(x):
    if pd.isna(x): return ""
    return f"{float(x):,.0f}".replace(",", " ").replace(".", ",")

def fmt_fr_2d(x):
    if pd.isna(x): return ""
    return f"{float(x):,.2f}".replace(",", " ").replace(".", ",")

zone_df_multi_fmt = zone_df_multi.copy()
for col in ["CA 2024 (€)", "Nb Visites", "Nb Clients"]:
    if col in zone_df_multi_fmt.columns:
        zone_df_multi_fmt[col] = zone_df_multi_fmt[col].apply(fmt_fr_int)
for col in ["ETP estimé", "Nb Visites / ETP", "Nb Clients / ETP"]:
    if col in zone_df_multi_fmt.columns:
        zone_df_multi_fmt[col] = zone_df_multi_fmt[col].apply(fmt_fr_2d)

st.dataframe(zone_df_multi_fmt, use_container_width=True)

# ------- EXPORT CSV (format FR : sep=';' + décimale=',') -------
csv_fr = zone_df_multi.copy()
csv_bytes = csv_fr.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")

st.download_button(
    "📥 Télécharger le tableau",
    data=csv_bytes,
    file_name="indicateurs_zones_multi_criteres.csv",
    mime="text/csv"
)

# Debug minimal
with st.expander("🔎 Debug (merged)"):
    st.write(merged[["Departement","lat","lon","CA 2024","Nb Visite","Nb Magasins","Zone"]].sort_values("Zone"))
