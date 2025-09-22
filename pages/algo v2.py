import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import folium
from streamlit_folium import st_folium
from sklearn.cluster import AgglomerativeClustering
 
st.set_page_config(layout="wide", page_title="Sectorisation interactive")
st.title("🗺️ Sectorisation automatique + modification manuelle")

departements_ile_de_france = {
    "75": "Paris",
    "77": "Seine-et-Marne",
    "78": "Yvelines",
    "91": "Essonne",
    "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne",
    "95": "Val-d'Oise",
}
IDF_SET = {d.zfill(2) for d in departements_ile_de_france.keys()}

# --- Upload fichier Excel ---
uploaded_file = st.sidebar.file_uploader("📂 Charger le fichier Excel des magasins", type=["xlsx"])
geojson_path = "geoson.geojson"

if uploaded_file is None:
    st.info("Veuillez charger un fichier Excel contenant les données magasins.")
    st.stop()
 
# --- Lecture des données ---
df = pd.read_excel(uploaded_file)
# 🧭 Normalisation Corse : 20 -> 2A/2B quand on peut (via Code Postal), sinon fallback 2A
# Unifier les départements en 20 côté Excel
# Normalise les noms de colonnes (trim, remplace espace insécable, etc.)
df.columns = (
    pd.Series(df.columns)
      .astype(str)
      .str.replace("\u00A0", " ", regex=False)  # remplace NBSP par espace normal
      .str.strip()
)

df["Departement"] = (
    df["Departement"].astype(str).str.strip().str.upper()
      .replace({"2A": "20", "2B": "20"})
      .str.zfill(2)
)

# df["Departement"] = df["Departement"].astype(str).str.upper().str.zfill(2)

# df["Departement"] = df["Departement"].astype(str).str.zfill(2)
df["Nb Visite"] = df["Nb Visite"].fillna(0)
df["CA 2024"] = df["CA 2024"].fillna(0)
df["Nb Magasins"] = 1
 
# Colonne facultative
client_name_col = "Nom du client" if "Nom du client" in df.columns else None
 
with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# ✅ Regrouper la Corse en "20" au lieu de 2A/2B
for feature in geojson_data["features"]:
    code = str(feature["properties"]["code"]).upper()
    if code in ["2A", "2B"]:
        feature["properties"]["code"] = "20"

gdf_dept = gpd.GeoDataFrame.from_features(geojson_data["features"]).set_crs("EPSG:4326")
gdf_dept["code"] = gdf_dept["code"].astype(str)
 
# --- Paramètres utilisateur ---
nb_clusters = st.sidebar.slider("🔢 Nombre de zones à générer", min_value=2, max_value=10, value=5)
diviseur_etp = st.sidebar.number_input("🔧 Valeur de référence pour ETP", value=949)
 
# ------------------------------------------------------------------------------------
# 🧹 Interface visuelle d'exclusion (table + recherche)
# ------------------------------------------------------------------------------------
st.sidebar.markdown("### 🧹 Exclusion de magasins")
 
# State init
if "excluded_clients" not in st.session_state:
    st.session_state.excluded_clients = []
if "last_excluded_set" not in st.session_state:
    st.session_state.last_excluded_set = set()
if "last_nb_clusters" not in st.session_state:
    st.session_state.last_nb_clusters = None
if "zones_modifiables" not in st.session_state:
    st.session_state.zones_modifiables = None

# --- State pour la variante IDF (édition manuelle à droite) ---
if "zones_modifiables_idf" not in st.session_state:
    st.session_state.zones_modifiables_idf = None
if "last_nb_clusters_idf" not in st.session_state:
    st.session_state.last_nb_clusters_idf = None
if "last_excluded_set_idf" not in st.session_state:
    st.session_state.last_excluded_set_idf = set()

# Table compactée par client
group_cols = ["Code du client"]
if client_name_col:
    group_cols.append(client_name_col)
 
clients_df = (
    df.assign(**{"Code du client": df["Code du client"].astype(str)})
      .groupby(group_cols, dropna=False)
      .agg({"Nb Visite": "sum", "CA 2024": "sum", "Departement": pd.Series.nunique})
      .reset_index()
      .rename(columns={"Departement": "Départements (nb distinct)"})
)
 
# Colonne de sélection
clients_df["Exclure"] = clients_df["Code du client"].astype(str).isin(st.session_state.excluded_clients)
 
# Recherche
search = st.sidebar.text_input("🔎 Rechercher (code ou nom)", value="")
if search:
    mask = clients_df["Code du client"].astype(str).str.contains(search, case=False, na=False)
    if client_name_col:
        mask |= clients_df[client_name_col].astype(str).str.contains(search, case=False, na=False)
    clients_view = clients_df[mask].copy()
else:
    clients_view = clients_df.copy()
 
# Editeur
edited_clients = st.data_editor(
    clients_view,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Exclure": st.column_config.CheckboxColumn(
            "Exclure", help="Cocher pour exclure ce magasin de tous les calculs."
        ),
        "Nb Visite": st.column_config.NumberColumn(format="%.0f"),
        "CA 2024": st.column_config.NumberColumn(format="%.2f"),
        "Départements (nb distinct)": st.column_config.NumberColumn(format="%.0f"),
    },
    key="editor_clients"
)
 
# Réconcilier les exclusions sur la base complète
if not edited_clients.empty:
    upd = edited_clients[["Code du client", "Exclure"]].copy()
    upd["Code du client"] = upd["Code du client"].astype(str)
    clients_df = clients_df.merge(upd, on="Code du client", how="left", suffixes=("", "_new"))
    clients_df["Exclure"] = np.where(clients_df["Exclure_new"].notna(), clients_df["Exclure_new"], clients_df["Exclure"])
    clients_df = clients_df.drop(columns=["Exclure_new"])
 
new_excluded = sorted(clients_df.loc[clients_df["Exclure"], "Code du client"].astype(str).unique().tolist())
 
# ⚠️ Si la liste d'exclus change: on reset le zonage et on relance
if set(new_excluded) != set(st.session_state.excluded_clients):
    st.session_state.excluded_clients = new_excluded
    st.session_state.zones_modifiables = None  # force recalcul du clustering
    st.rerun()
 
st.sidebar.caption(f"Magasins exclus : {len(st.session_state.excluded_clients)} / {clients_df['Code du client'].nunique()}")
 
# Appliquer l'exclusion au DF de travail
df_f = df[~df["Code du client"].astype(str).isin(st.session_state.excluded_clients)].copy()
# Cherche la colonne "région" avec plusieurs alias possibles
# --- Détection robuste de la colonne région (gère "Région admnistrative") ---
REGION_CANDIDATES = [
    "Région admnistrative", "Region admnistrative",  # <- avec la faute
    "Région administrative", "Region administrative",
    "Région", "Region"
]
REGION_COL = next((c for c in REGION_CANDIDATES if c in df_f.columns), None)
st.caption(f"Colonne région détectée : {REGION_COL or 'Aucune'}")

# Colonne région "canonique" propre (pour éviter les soucis d’espaces/nbsp)
if REGION_COL:
    df_f["__REGION_CANON__"] = (
        df_f[REGION_COL]
        .astype(str)
        .str.replace("\u00A0", " ", regex=False)  # NBSP -> espace normal
        .str.replace(r"\s+", " ", regex=True)     # espaces multiples -> un espace
        .str.strip()
    )
else:
    df_f["__REGION_CANON__"] = np.nan

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# 📍 Calcul des centroïdes (robuste, et Corse=20)
# ------------------------------------------------------------------------------------
# On part du gdf_dept déjà créé plus haut
gdf_dept["Departement"] = (
    gdf_dept["code"].astype(str).str.strip().str.upper()
            .replace({"2A": "20", "2B": "20"})
            .str.zfill(2)
)

# Dissoudre pour n'avoir qu'UNE ligne par département (la Corse = 20 unique)
gdf_diss = gdf_dept.dissolve(by="Departement", as_index=True)

# Centroïdes robustes (Polygon/MultiPolygon)
centroids = gdf_diss.geometry.centroid
centroids_df = pd.DataFrame({
    "Departement": gdf_diss.index,
    "lon": centroids.x.values,
    "lat": centroids.y.values
}).reset_index(drop=True)


# ------------------------------------------------------------------------------------
# 🧮 Agrégation par département (APRES EXCLUSION)
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# 🧮 Agrégation par département (APRES EXCLUSION)
# ------------------------------------------------------------------------------------
dept_data = df_f.groupby("Departement").agg({
    "Nb Magasins": "sum",
    "Nb Visite": "sum",
    "CA 2024": "sum"
}).reset_index()

merged = pd.merge(dept_data, centroids_df, on="Departement", how="left").dropna(subset=["lat", "lon"])

# 🔎 Diagnostic : voir quels départements n'ont pas trouvé de centroïde
missing_after_merge = sorted(set(dept_data["Departement"]) - set(merged["Departement"]))
if missing_after_merge:
    st.warning(f"Départements sans centroïde (non mergés) : {', '.join(missing_after_merge)}")
# 🔁 Si la liste des départements disponibles a changé (ex : 2A/2B -> 20), on force un nouveau clustering
if st.session_state.zones_modifiables is not None:
    depts_current = sorted(merged["Departement"].unique().tolist())
    depts_cached = sorted({d for deps in st.session_state.zones_modifiables.values() for d in deps})
    if depts_current != depts_cached:
        st.session_state.zones_modifiables = None
# ------------------------------------------------------------------------------------
# 🔀 Refaire la sectorisation (clustering) si nécessaire
#    - toujours refait si exclusions changent (zones_modifiables is None ci-dessus)
#    - refait si nombre de clusters change
# ------------------------------------------------------------------------------------
need_recluster = (
    st.session_state.zones_modifiables is None
    or nb_clusters != st.session_state.last_nb_clusters
    or set(st.session_state.excluded_clients) != st.session_state.last_excluded_set
)
 
if need_recluster:
    if len(merged) >= max(1, nb_clusters):
        geo_features = merged[["lat", "lon"]].to_numpy()
        agglo = AgglomerativeClustering(n_clusters=nb_clusters, linkage="ward")
        cluster_labels = agglo.fit_predict(geo_features)
 
        # Re-normaliser les labels
        unique_labels = sorted(np.unique(cluster_labels))
        label_map = {old: i for i, old in enumerate(unique_labels)}
        normalized_labels = np.array([label_map[l] for l in cluster_labels])
 
        merged["Cluster"] = normalized_labels
        merged["Zone"] = merged["Cluster"].apply(lambda x: f"Zone {chr(65 + x)}")
 
        # Zone → liste de départements
        zones_dict = merged.groupby("Zone")["Departement"].apply(list).to_dict()
        st.session_state.zones_modifiables = zones_dict
        st.session_state.last_nb_clusters = nb_clusters
        st.session_state.last_excluded_set = set(st.session_state.excluded_clients)
 
        st.success("✅ Sectorisation et zonage recalculés suite aux exclusions/modifications.")
    else:
        st.warning("Pas assez de départements pour effectuer le clustering avec le nombre de zones demandé.")
        st.stop()
 
# ------------------------------------------------------------------------------------
# 🗺️ Carte interactive

# --- 🔢 Indicateurs Clés ---
st.subheader("Indicateurs Clés")

# Table Zones fiable à 100% (indépendante de 'merged')
zones_df = pd.DataFrame(
    [{"Departement": dep, "Zone": zone}
     for zone, deps in st.session_state.zones_modifiables.items()
     for dep in deps]
)

# On exclut les lignes sans zone attribuée
df_sectorised = df.merge(zones_df, on="Departement", how="left")
df_sectorised = df_sectorised[df_sectorised["Zone"].notna()]

# --- ⚠️ Magasins non sectorisés ---
with st.sidebar.expander("⚠️ Magasins non sectorisés", expanded=True):
    excluded_depts = df[~df["Departement"].isin(zones_df["Departement"])]

    if excluded_depts.empty:
        st.success("✅ Tous les magasins ont été assignés à une zone.")
    else:
        nb_mag_exclu = len(excluded_depts)
        st.warning(f"""
        🚫 **{nb_mag_exclu} magasin(s)** n'ont pas été assignés à une zone de sectorisation.
        ℹ️ **Raison** : DROM-COM / hors périmètre / codes spéciaux, etc.
        """)

        region_col = "Région" if "Région" in excluded_depts.columns else ("Region" if "Region" in excluded_depts.columns else None)
        if region_col:
            excl_summary = (excluded_depts.groupby(["Departement", region_col])
                            .agg({"Nb Magasins":"sum","Nb Visite":"sum","CA 2024":"sum"})
                            .reset_index().sort_values(by="Nb Magasins", ascending=False))
        else:
            excl_summary = (excluded_depts.groupby("Departement")
                            .agg({"Nb Magasins":"sum","Nb Visite":"sum","CA 2024":"sum"})
                            .reset_index().sort_values(by="Nb Magasins", ascending=False))

        st.markdown("### 🧾 Détails par département non sectorisé :")
        st.dataframe(excl_summary, use_container_width=True)

        st.download_button(
            "📥 Télécharger les magasins exclus",
            data=excluded_depts.to_csv(index=False).encode("utf-8"),
            file_name="magasins_non_sectorises.csv",
            mime="text/csv"
        )

# Calculs globaux
nb_magasins_total = len(df_sectorised)
nb_visites_total = df["Nb Visite"].sum()
ca_total_2024 = df["CA 2024"].sum()
etp_total = round(nb_visites_total / diviseur_etp, 2)


# 💅 CSS des cards
st.markdown("""
<style>
.card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    margin: 10px;
}
.card h2 {
    font-size: 2rem;
    margin: 0;
}
.card p {
    margin: 5px 0;
    font-size: 1.2rem;
    color: #333;
}
.card .delta {
    font-size: 1rem;
    margin-top: 5px;
    color: green;
}
</style>
""", unsafe_allow_html=True)

# Affichage en 4 cards
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="card">
        <h2>{nb_magasins_total}</h2>
        <p>Magasins couverts</p>
        <div class="delta">100%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h2>{int(nb_visites_total)}</h2>
        <p>Nb total de visites</p>
        <div class="delta">100%</div>
    </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown(f"""
    <div class="card">
        <h2>{ca_total_2024:,.0f} €</h2>
        <p>CA total 2024</p>
        <div class="delta">100%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="card">
        <h2>{etp_total}</h2>
        <p>ETP estimés</p>
        <div class="delta">100%</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# 1) Clustering STANDARD (tous départements)
# =========================
# # (recalcule un clustering sur 'merged' pour avoir zones_standard)
# geo_all = merged[["lat", "lon"]].to_numpy()
# agg_std = AgglomerativeClustering(n_clusters=nb_clusters, linkage="ward")
# labels_std = agg_std.fit_predict(geo_all)

# # normalise les labels pour avoir Zone A, B, C...
# unique_std = sorted(np.unique(labels_std))
# map_std = {old: i for i, old in enumerate(unique_std)}  # 0..K-1
# merged_std = merged.copy()
# merged_std["Cluster"] = [map_std[l] for l in labels_std]
# merged_std["Zone"] = merged_std["Cluster"].apply(lambda x: f"Zone {chr(65 + x)}")

# zones_standard = merged_std.groupby("Zone")["Departement"].apply(list).to_dict()
# Après zones_standard = merged_std.groupby("Zone")["Departement"].apply(list).to_dict()
# Palette unique pour les deux cartes (mêmes couleurs)
palette = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
           "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
           "#008080", "#e6beff"]

# Map "Zone A/B/C..." -> couleur (A=palette[0], B=palette[1], etc.)
zone_color_map = {f"Zone {chr(65+i)}": palette[i % len(palette)] for i in range(nb_clusters)}

# Map departement -> zone (celle de la CARTE DE GAUCHE)
dept_to_zone_std = {
    dep: zone
    for zone, deps in st.session_state.zones_modifiables.items()
    for dep in deps
}

# Map departement -> couleur (hérite de la couleur de sa zone)
dept_to_color_std = {}
for dep, z in dept_to_zone_std.items():
    dept_to_color_std[dep] = zone_color_map.get(z, "#d9d9d9")


# =========================
# 2) Clustering HORS IDF + ajout d’une Zone "Région Île-de-France"
# =========================
merged["Departement"] = merged["Departement"].astype(str).str.zfill(2)
merged_idf = merged[merged["Departement"].isin(IDF_SET)].copy()
merged_rest = merged[~merged["Departement"].isin(IDF_SET)].copy()

zones_idf = {}
if len(merged_rest) >= max(1, nb_clusters):
    geo_features = merged_rest[["lat", "lon"]].to_numpy()
    agg_idf = AgglomerativeClustering(n_clusters=nb_clusters, linkage="ward")
    labels_idf = agg_idf.fit_predict(geo_features)

    unique_idf = sorted(np.unique(labels_idf))
    map_idf = {old: i for i, old in enumerate(unique_idf)}
    merged_rest = merged_rest.copy()
    merged_rest["Cluster"] = [map_idf[l] for l in labels_idf]
    merged_rest["Zone"] = merged_rest["Cluster"].apply(lambda x: f"Zone {chr(65 + x)}")

    zones_idf = merged_rest.groupby("Zone")["Departement"].apply(list).to_dict()

    # Donner à l’IDF un nom cohérent avec les autres zones (après Zone A..)
    if not merged_idf.empty:
        # lettre suivante après les zones A.. déjà créées (len(zones_idf) == nb_clusters)
        next_letter = chr(65 + len(zones_idf))     # ex: 5 -> 'F'
        idf_label = f"Zone {next_letter}"
        zones_idf[idf_label] = sorted(merged_idf["Departement"].unique().tolist())

else:
    st.warning("Pas assez de départements (hors IDF) pour clusteriser la variante IDF.")

# --- Alimente l'état pour IDF si nécessaire (recluster / exclusions / nb_clusters changent) ---
recluster_idf = (
    st.session_state.zones_modifiables_idf is None
    or nb_clusters != st.session_state.last_nb_clusters_idf
    or set(st.session_state.excluded_clients) != st.session_state.last_excluded_set_idf
)

if recluster_idf:
    st.session_state.zones_modifiables_idf = {z: [str(d).zfill(2) for d in deps] for z, deps in zones_idf.items()}
    st.session_state.last_nb_clusters_idf = nb_clusters
    st.session_state.last_excluded_set_idf = set(st.session_state.excluded_clients)

col1, col2 = st.columns(2)
with col1:
    # ------------------------------------------------------------------------------------
    st.subheader("🗺️ Carte des départements sectorisés")
    
    # Attribution de la zone à chaque feature GeoJSON
    for feature in geojson_data["features"]:
        code_dep = str(feature["properties"]["code"]).strip().upper().zfill(2)
        zone_name = next((z for z, deps in st.session_state.zones_modifiables.items() if code_dep in deps), None)
        feature["properties"]["Zone"] = zone_name or "Non assigné"
    
    m = folium.Map(location=[46.7, 2.5], zoom_start=6)
    folium.GeoJson(
        geojson_data,
        name="Départements",
        style_function=lambda feature: {
            'fillColor': zone_color_map.get(feature["properties"]["Zone"], "#d9d9d9"),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["nom", "code", "Zone"],
            aliases=["Nom", "Code", "Zone"]
        )
    ).add_to(m)
    
    st_folium(m, width=1000, height=600)
with col2:
    st.markdown("#### Carte avec IDF séparée")
    import copy
    geojson_idf = copy.deepcopy(geojson_data)

    # Attribuer le NOM de zone (variante IDF) depuis l'état
    for feature in geojson_idf["features"]:
        code_dep = str(feature["properties"]["code"]).strip().upper().zfill(2)
        zone_name_idf = next(
            (z for z, deps in st.session_state.zones_modifiables_idf.items() if code_dep in deps),
            None
        )
        feature["properties"]["Zone_IDF"] = zone_name_idf or "Non assigné"

    # Palette propre à la carte IDF (une couleur par zone IDF)
    zone_names_idf = list((st.session_state.zones_modifiables_idf or zones_idf).keys())
    zone_color_map_idf = {z: palette[i % len(palette)] for i, z in enumerate(zone_names_idf)}


    # Couleur = couleur standard par département (sauf IDF en or)
    def style_idf(feature):
        z = feature["properties"].get("Zone_IDF")
        return {
            'fillColor': zone_color_map_idf.get(z, "#d9d9d9"),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        }


    m2 = folium.Map(location=[46.7, 2.5], zoom_start=6)
    folium.GeoJson(
        geojson_idf,
        name="Départements",
        style_function=style_idf,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom", "code", "Zone_IDF"],
            aliases=["Nom", "Code", "Zone"]
        )
    ).add_to(m2)

    st_folium(m2, width=1000, height=600)

# ---------- Helper: construit le tableau d'indicateurs pour un découpage ----------
def build_zone_table(zones_dict, df_base, diviseur, region_col=None):
    rows = []
    seen = set()  # évite le double comptage des clients à l’intérieur d’un découpage
    df_base = df_base.copy()
    df_base["Departement"] = df_base["Departement"].astype(str).str.zfill(2)
    if "Code du client" in df_base.columns:
        df_base["Code du client"] = df_base["Code du client"].astype(str)

    for zone, deps in zones_dict.items():
        deps_norm = [str(d).zfill(2) for d in deps]
        zdf = df_base[df_base["Departement"].isin(deps_norm)].copy()

        if "Code du client" in zdf.columns:
            zdf = zdf[~zdf["Code du client"].isin(seen)]
            seen.update(zdf["Code du client"])

        ca = float(zdf["CA 2024"].sum())
        visites = float(zdf["Nb Visite"].sum())
        nb_magasins = int(zdf["Code du client"].nunique()) if "Code du client" in zdf.columns else int(zdf.shape[0])
        etp = visites / diviseur if diviseur > 0 else 0.0

        # --- Régions admin pour la zone ---
        # --- Régions admin pour la zone ---
        if region_col and region_col in zdf.columns:
            regs = (
                zdf[region_col]
                .dropna()
                .astype(str)
                .str.replace("\u00A0", " ", regex=False)  # NBSP -> espace normal
                .str.replace(r"\s+", " ", regex=True)     # espaces multiples -> un espace
                .str.strip()
                .tolist()
            )
            regs = sorted({r for r in regs if r})  # uniques & triées
            regions_list = " · ".join(regs)
            # nb_regions = len(regs)
        else:
            regions_list, nb_regions = "", 0



        rows.append({
            "Zone": zone,
            "Départements": ", ".join(sorted(deps_norm)),
            # "Régions (nb)": nb_regions,
            "Régions (liste)": regions_list,
            "Magasins": nb_magasins,
            "CA 2024 (€)": ca,
            "Nb Visites": visites,
            "ETP estimé": round(etp, 2),
            "Nb Visites / ETP": round(visites / etp, 2) if etp > 0 else 0.0,
            "Nb Clients / ETP": round(nb_magasins / etp, 2) if etp > 0 else 0.0
        })

    df_out = pd.DataFrame(rows).sort_values("Zone")
    # (Optionnel) ordre de colonnes agréable
    wanted = ["Zone","Départements","Régions (nb)","Régions (liste)","Magasins",
              "CA 2024 (€)","Nb Visites","ETP estimé","Nb Visites / ETP","Nb Clients / ETP"]
    return df_out[[c for c in wanted if c in df_out.columns]]

# ---------- Indicateurs par RÉGION et par ZONE (avec /ETP) ----------
def build_zone_region_table(zones_dict, df_base, diviseur, region_col):
    if not region_col or region_col not in df_base.columns:
        return pd.DataFrame()

    df_base = df_base.copy()
    df_base["Departement"] = df_base["Departement"].astype(str).str.zfill(2)
    df_base[region_col] = (
        df_base[region_col].astype(str)
        .str.replace("\u00A0", " ", regex=False)   # NBSP -> espace normal
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    has_client_col = "Code du client" in df_base.columns
    if has_client_col:
        df_base["Code du client"] = df_base["Code du client"].astype(str)

    all_rows = []
    for zone, deps in zones_dict.items():
        deps_norm = [str(d).zfill(2) for d in deps]
        zdf = df_base[df_base["Departement"].isin(deps_norm)].copy()
        if zdf.empty:
            continue

        # pas de double comptage de client à l’intérieur de la zone
        if has_client_col:
            zdf = zdf.drop_duplicates(subset=["Code du client"])

        if has_client_col:
            g = zdf.groupby(region_col, dropna=False).agg(
                **{
                    "Nb Clients": ("Code du client", "nunique"),
                    "CA 2024 (€)": ("CA 2024", "sum"),
                    "Nb Visites": ("Nb Visite", "sum"),
                }
            ).reset_index(names=["Région"])
        else:
            g = zdf.groupby(region_col, dropna=False).agg(
                **{
                    "CA 2024 (€)": ("CA 2024", "sum"),
                    "Nb Visites": ("Nb Visite", "sum"),
                    "Nb Clients": ("Departement", "size"),
                }
            ).reset_index(names=["Région"])

        # ETP et ratios
        if diviseur > 0:
            g["ETP estimé"] = (g["Nb Visites"] / diviseur).round(2)
            g["Nb Visites / ETP"] = (g["Nb Visites"] / g["ETP estimé"]).replace([np.inf, -np.inf], 0).fillna(0).round(2)
            g["Nb Clients / ETP"] = (g["Nb Clients"] / g["ETP estimé"]).replace([np.inf, -np.inf], 0).fillna(0).round(2)
        else:
            g["ETP estimé"] = 0.0
            g["Nb Visites / ETP"] = 0.0
            g["Nb Clients / ETP"] = 0.0

        g.insert(0, "Zone", zone)
        all_rows.append(g)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True).sort_values(["Zone", "Région"])
    cols = ["Zone", "Région", "Nb Clients", "CA 2024 (€)", "Nb Visites",
            "ETP estimé", "Nb Visites / ETP", "Nb Clients / ETP"]
    return out[cols]


col1, col2 = st.columns(2)

with col1:
    # ... (votre code folium m)
    st_folium(m, width=1000, height=600)

    # ⬇️ Tableau sous la carte standard
    st.markdown("#### Indicateurs par zone — carte standard")
    zone_df_std = build_zone_table(
        st.session_state.zones_modifiables, df_f, diviseur_etp, region_col="__REGION_CANON__"
    ).rename(columns={
        "Magasins": "Nb Clients",
        "CA 2024 (€)": "CA 2024 (€)"
    })

    st.dataframe(zone_df_std, use_container_width=True)
    st.download_button(
        "📥 Télécharger (standard)",
        data=zone_df_std.to_csv(index=False).encode("utf-8"),
        file_name="indicateurs_zones_standard.csv",
        mime="text/csv"
    )
    # ----- Détail par RÉGION (carte standard) -----
    st.markdown("##### Indicateurs par région — carte standard")
    region_df_std = build_zone_region_table(st.session_state.zones_modifiables, df_f, diviseur_etp, REGION_COL)
    if region_df_std.empty:
        st.info("Aucune colonne région trouvée dans l’Excel (ex. « Région administrative »).")
    else:
        st.dataframe(region_df_std, use_container_width=True)
        st.download_button(
            "📥 Télécharger (régions · standard)",
            data=region_df_std.to_csv(index=False).encode("utf-8"),
            file_name="indicateurs_regions_standard.csv",
            mime="text/csv",
        )

with col2:
    # ... (votre code folium m2)
    st_folium(m2, width=1000, height=600)

    # ⬇️ Tableau sous la carte IDF
    st.markdown("#### Indicateurs par zone — IDF séparée")
    zone_df_idf = build_zone_table(
        st.session_state.zones_modifiables_idf or zones_idf,  # fallback si jamais None
        df_f, diviseur_etp, region_col=REGION_COL
    ).rename(columns={
        "Magasins": "Nb Clients",
        "CA 2024 (€)": "CA 2024 (€)"
    })

    st.dataframe(zone_df_idf, use_container_width=True)
    st.download_button(
        "📥 Télécharger (IDF séparée)",
        data=zone_df_idf.to_csv(index=False).encode("utf-8"),
        file_name="indicateurs_zones_idf.csv",
        mime="text/csv"
    )
    # ----- Détail par RÉGION (carte IDF séparée) -----
    st.markdown("##### Indicateurs par région — IDF séparée")
    region_df_idf = build_zone_region_table(
    st.session_state.zones_modifiables_idf or zones_idf,  # ⬅️ prend le zonage édité, fallback si None
    df_f,
    diviseur_etp,
    REGION_COL
    )
    if region_df_idf.empty:
        st.info("Aucune colonne région trouvée dans l’Excel (ex. « Région administrative »).")
    else:
        st.dataframe(region_df_idf, use_container_width=True)
        st.download_button(
            "📥 Télécharger (régions · IDF séparée)",
            data=region_df_idf.to_csv(index=False).encode("utf-8"),
            file_name="indicateurs_regions_idf.csv",
            mime="text/csv",
        )

 
# ------------------------------------------------------------------------------------
# 🛠 Modification manuelle des zones
# ------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🛠 Modifier l’affectation des départements")
 
all_departments = sorted(set(dep for deps in st.session_state.zones_modifiables.values() for dep in deps))
departements_to_move = st.multiselect("Départements à déplacer :", all_departments)
new_zone = st.selectbox("Nouvelle zone :", list(st.session_state.zones_modifiables.keys()))
 
if st.button("✅ Réassigner"):
    moved = []
    for dep in departements_to_move:
        for z, deps in st.session_state.zones_modifiables.items():
            if dep in deps:
                st.session_state.zones_modifiables[z].remove(dep)
                break
        st.session_state.zones_modifiables[new_zone].append(dep)
        moved.append(dep)
    if moved:
        st.success(f"✅ {', '.join(moved)} déplacé(s) vers {new_zone}.")
        st.rerun()
    else:
        st.warning("Aucun département déplacé.")


st.markdown("---")
st.markdown("### 🛠 Modifier l’affectation des départements (carte IDF séparée)")

# 100% des départements présents dans le zonage IDF courant (y compris l’IDF)
all_deps_idf = sorted({d for deps in st.session_state.zones_modifiables_idf.values() for d in deps})

deps_to_move_idf = st.multiselect(
    "Départements à déplacer (IDF séparée) :",
    all_deps_idf,
    key="move_deps_idf"
)

dest_zone_idf = st.selectbox(
    "Nouvelle zone (IDF séparée) :",
    list(st.session_state.zones_modifiables_idf.keys()),
    key="dest_zone_idf"
)

if st.button("✅ Réassigner (IDF séparée)"):
    moved = []
    for dep in deps_to_move_idf:
        # Retirer le département de sa zone actuelle
        for z, deps in st.session_state.zones_modifiables_idf.items():
            if dep in deps:
                deps.remove(dep)
                break
        # L'ajouter dans la zone cible
        st.session_state.zones_modifiables_idf[dest_zone_idf].append(dep)
        moved.append(dep)

    if moved:
        st.success(f"✅ {', '.join(moved)} déplacé(s) vers {dest_zone_idf}.")
        st.rerun()
    else:
        st.warning("Aucun département déplacé.")

 
# ------------------------------------------------------------------------------------
# 📤 Exports
# ------------------------------------------------------------------------------------
st.markdown("### 📤 Exports")
 
# Export de la sectorisation (Département → Zone)
final_export = []
for zone, depts in st.session_state.zones_modifiables.items():
    for dep in depts:
        final_export.append({"Departement": dep, "Zone": zone})
export_df = pd.DataFrame(final_export)
st.download_button("📥 Télécharger la sectorisation (départements-zones)",
                   data=export_df.to_csv(index=False), file_name="sectorisation_finale.csv")
 