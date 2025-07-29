import streamlit as st
import pandas as pd
import folium
from folium import Choropleth, GeoJson, GeoJsonTooltip
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import math
from shapely.geometry import Point
import geopandas as gpd

st.set_page_config(layout="wide")
st.title("📍 Sectorisation automatique par département")

# --- Chargement des données ---
uploaded_file = st.sidebar.file_uploader("📂 Charger le fichier Excel avec les données magasins", type=["xlsx"])
geojson_file = "geoson.geojson"  # fichier GeoJSON local des départements (code_insee)

if uploaded_file is not None and geojson_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("Fichier chargé avec succès !")
    st.sidebar.write(f"Nombre total de lignes dans le fichier brut : {len(df)}")

    # Charger le GeoJSON
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)

    # --- Étape 1 : Convertir en GeoDataFrame
    gdf_mag = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["long"], df["lat"]),
        crs="EPSG:4326"
    )

    # --- Étape 2 : Charger GeoJSON avec codes 2A / 2B
    gdf_dept = gpd.GeoDataFrame.from_features(geojson_data["features"])
    gdf_dept = gdf_dept.set_crs("EPSG:4326")  # CRS pour matcher avec les points
    # Harmoniser les formats pour la fusion (code string vs. int)
    gdf_dept["code"] = gdf_dept["code"].astype(str)

    df["Departement"] = df["Departement"].astype(str).str.strip()

    # Gestion spéciale Corse
    df["Departement"] = df["Departement"].replace({"20": "2A"})  # ou logique plus avancée si tu veux distinguer 2A/2B
    # Remplissage à gauche pour uniformiser (1 → 01, 2 → 02, etc.)
    df["Departement"] = df["Departement"].apply(lambda x: x if x in ["2A", "2B"] else x.zfill(2))


    # --- Étape 5 : Nettoyage & suite du pipeline
    df["Nb Visite"] = df["Nb Visite"].fillna(0)
    df["CA 2023"] = df["CA 2023"].fillna(0)

    df["Nb Magasins"] = 1

    
    # Agréger par département
    dept_data = df.groupby("Departement").agg({
        "Nb Magasins": "sum",
        "Nb Visite": "sum",
        "CA 2023": "sum"
    }).reset_index()

    # 1. Extraire les centroïdes des départements
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)
    centroids = []
    codes = []
    for feature in geojson_data["features"]:
        code = feature["properties"]["code"]
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            coords = np.array(geometry["coordinates"][0])
        else:  # MultiPolygon
            coords = np.array(geometry["coordinates"][0][0])
        centroid = coords.mean(axis=0)
        centroids.append(centroid)
        codes.append(code)

    # 2. Associer centroïdes aux départements
    centroids_df = pd.DataFrame(centroids, columns=["lon", "lat"])
    centroids_df["Departement"] = codes

    # 3. Fusionner avec les données
    merged = pd.merge(dept_data, centroids_df, on="Departement", how="left")
    merged = merged.dropna(subset=["lat", "lon"])

    # 4. Clustering hiérarchique basé sur la géographie
    geo_features = merged[["lat", "lon"]].to_numpy()
    agglo = AgglomerativeClustering(n_clusters=5, linkage="ward")
    merged["Zone"] = agglo.fit_predict(geo_features)

    # 5. Mise à jour des zones dans dept_data
    if "Zone" in dept_data.columns:
        dept_data = pd.merge(dept_data.drop("Zone", axis=1), merged[["Departement", "Zone"]], on="Departement", how="left")
    else:
        dept_data = pd.merge(dept_data, merged[["Departement", "Zone"]], on="Departement", how="left")


    # Mapping pour couleurs
    zone_colors = {
        "Zone A": "red",
        "Zone B": "blue",
        "Zone C": "green",
        "Zone D": "orange",
        "Zone E": "violet"
    }

    dept_data["Color"] = dept_data["Zone"].map(zone_colors)

    # Charger le fichier GeoJSON
    with open(geojson_file, encoding="utf-8") as f:
        geojson_data = json.load(f)

    # Ajouter les données de zone aux features du GeoJSON
    for feature in geojson_data["features"]:
        code_dep = str(feature["properties"]["code"]).strip()  # S'assurer que c'est une string bien formatée
        row = dept_data[dept_data["Departement"] == code_dep]
        if not row.empty:
            feature["properties"]["Zone"] = f"Zone {chr(65 + int(row['Zone'].values[0]))}"
            feature["properties"]["CA"] = int(row["CA 2023"].values[0])
        else:
            feature["properties"]["Zone"] = "Non défini"
            feature["properties"]["CA"] = 0
    colA, colB = st.columns(2)
    # Partie gauche (col1)
    with colA:
        # --- 🔢 Indicateurs Clés ---
        st.subheader("Indicateurs Clés")

        # ETP de référence (modifiable)
        diviseur_etp = 949  # Tu peux mettre 230 ou autre valeur métier
        # On exclut les lignes sans zone attribuée (comme "98" → "Zone ?")
        df_sectorised = pd.merge(df, dept_data[["Departement", "Zone"]], on="Departement", how="left")
        df_sectorised = df_sectorised[df_sectorised["Zone"].notna()]
        # --- ⚠️ Magasins non sectorisés ---
        with st.sidebar.expander("⚠️ Magasins non sectorisés", expanded=True):
            excluded_depts = df[~df["Departement"].isin(df_sectorised["Departement"])]

            if excluded_depts.empty:
                st.success("✅ Tous les magasins ont été assignés à une zone.")
            else:
                nb_mag_exclu = len(excluded_depts)
                st.warning(f"""
                🚫 **{nb_mag_exclu} magasin(s)** n'ont pas été assignés à une zone de sectorisation.
                
                ℹ️ **Raison** : Ces magasins sont situés dans des départements qui ne font pas partie de la France métropolitaine, ou absents du périmètre de sectorisation (ex : DROM-COM, codes spéciaux...).
                """)

                # Vérifie la présence de la colonne "Région"
                region_col = "Région" if "Région" in excluded_depts.columns else ("Region" if "Region" in excluded_depts.columns else None)

                if region_col:
                    excl_summary = excluded_depts.groupby(["Departement", region_col]).agg({
                        "Nb Magasins": "sum",
                        "Nb Visite": "sum",
                        "CA 2023": "sum"
                    }).reset_index().sort_values(by="Nb Magasins", ascending=False)
                else:
                    excl_summary = excluded_depts.groupby("Departement").agg({
                        "Nb Magasins": "sum",
                        "Nb Visite": "sum",
                        "CA 2023": "sum"
                    }).reset_index().sort_values(by="Nb Magasins", ascending=False)

                st.markdown("### 🧾 Détails par département non sectorisé :")
                st.dataframe(excl_summary, use_container_width=True)

                st.download_button(
                    label="📥 Télécharger les magasins exclus",
                    data=excluded_depts.to_csv(index=False).encode("utf-8"),
                    file_name="magasins_non_sectorises.csv",
                    mime="text/csv"
                )

        # Calculs
        # nb_magasins_total = df["Code du client"].nunique()
        # nb_magasins_total = len(df)
        nb_magasins_total = len(df_sectorised)
        nb_visites_total = df["Nb Visite"].sum()
        ca_total_2023 = df["CA 2023"].sum()
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
                <h2>{ca_total_2023:,.0f} €</h2>
                <p>CA total 2023</p>
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
            # === Tableau 1 : Nombre de magasins et CA total par département ===
        table1 = df.groupby("Departement").agg(
            Nombre_Magasins=('Nb Magasins', 'sum'),
            Total_CA_2023=('CA 2023', 'sum')
        ).reset_index()

        # st.dataframe(table1.style.format({"Total_CA_2023": "{:,.2f}"}), use_container_width=True)
        # === Tableau 2 : Synthèse par zone ===
        # Convertir code Zone en A/B/C...
        dept_data["Nom_Zone"] = dept_data["Zone"].apply(lambda x: f"Zone {chr(65 + int(x))}" if pd.notnull(x) else "Zone ?")

        # Regrouper les départements par zone
        zone_summary = dept_data.groupby("Nom_Zone").agg({
            "Departement": lambda x: ', '.join(sorted(x)),
            "Nb Magasins": "sum",
            "CA 2023": "sum",
            "Nb Visite": "sum"
        }).reset_index()

        # Ajouter colonne ETP
        zone_summary["ETP"] = (zone_summary["Nb Visite"] / diviseur_etp).round(2)

        # Renommer les colonnes
        zone_summary.columns = ["Zone", "Départements", "Nombre de Magasins", "Total CA (€)", "Nb Visites", "ETP"]

        # Affichage
        # st.dataframe(zone_summary.style.format({"Total CA (€)": "{:,.2f}"}), use_container_width=True)
        with st.expander("📄 Détails statistiques par département et par zone", expanded=True):
            st.markdown("#### Par département")
            st.dataframe(table1.style.format({"Total_CA_2023": lambda x: f"{x:,.0f}".replace(",", " ")}), use_container_width=True)

            st.markdown("#### Par zone")
            st.dataframe(zone_summary.style.format({
                "Total CA (€)": lambda x: f"{x:,.0f}".replace(",", " "),
                "ETP": "{:.2f}"
            }), use_container_width=True)


    # Partie droite (col2)  
    with colB:
        # Création de la carte
        m = folium.Map(location=[46.7, 2.5], zoom_start=6, tiles="cartodbpositron")

        # Ajout des polygones
        GeoJson(
            geojson_data,
            style_function=lambda feature: {
                "fillColor": zone_colors.get(feature["properties"]["Zone"], "#d9d9d9"),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.6
            },

            tooltip=GeoJsonTooltip(fields=["nom", "Zone", "CA"], aliases=["Département", "Zone", "CA total"])
        ).add_to(m)

        # Affichage dans Streamlit
        st.markdown("### Carte des départements sectorisés automatiquement")
        st_data = st_folium(m, width=1000)

        # Export CSV
        csv_export = dept_data.copy()
        csv_export["Zone"] = csv_export["Zone"].apply(
            lambda x: f"Zone {chr(65 + int(x))}" if pd.notnull(x) and not math.isnan(x) else "Zone ?"
        )   
        st.download_button("📥 Télécharger la sectorisation", data=csv_export.to_csv(index=False), file_name="sectorisation_par_departement.csv")

else:
    st.info("📎 Veuillez charger un fichier Excel et vérifier que le fichier GeoJSON `departements.geojson` est bien présent.")
