import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import json
from sklearn.cluster import AgglomerativeClustering
import folium
from streamlit_folium import st_folium
import string

# Configuration de la page
st.set_page_config(
    page_title="Sectorisation minimale",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clean_column_names(df):
    """Nettoie les noms de colonnes en remplaçant NBSP et en trimant"""
    df.columns = df.columns.str.replace('\xa0', ' ').str.strip()
    return df

def normalize_department_code(dept_code):
    """Normalise les codes départements"""
    if pd.isna(dept_code):
        return None
    
    dept_str = str(dept_code).upper().strip()
    
    # Normaliser la Corse
    if dept_str in ["2A", "2B"]:
        dept_str = "20"
    
    # DOM-TOM (971-976) : laisser tel quel si déjà 3 chiffres
    if dept_str in ["971", "972", "973", "974", "975", "976"]:
        return dept_str
    
    # Autres départements : zfill(2)
    try:
        dept_num = int(dept_str)
        if dept_num < 100:
            return str(dept_num).zfill(2)
        else:
            return dept_str
    except:
        return dept_str

def load_and_process_data(uploaded_file):
    """Charge et traite le fichier Excel"""
    try:
        # Lire le fichier Excel
        df = pd.read_excel(uploaded_file)
        df = clean_column_names(df)
        
        # Vérifier les colonnes requises
        required_cols = ['Code du client', 'Departement', 'Nb Visite', 'CA 2024']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonnes manquantes : {missing_cols}")
            return None
        
        # Nettoyer les données
        df = df.dropna(subset=['Code du client', 'Departement'])
        df['Departement'] = df['Departement'].apply(normalize_department_code)
        df = df.dropna(subset=['Departement'])
        
        # Convertir les types
        df['Nb Visite'] = pd.to_numeric(df['Nb Visite'], errors='coerce').fillna(0)
        df['CA 2024'] = pd.to_numeric(df['CA 2024'], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def load_geojson():
    """Charge le GeoJSON des départements"""
    try:
        gdf = gpd.read_file("geoson.geojson")
        
        # Normaliser les codes départements dans le GeoJSON
        if 'code' in gdf.columns:
            gdf['code'] = gdf['code'].apply(normalize_department_code)
        elif 'CODE_DEPT' in gdf.columns:
            gdf['code'] = gdf['CODE_DEPT'].apply(normalize_department_code)
        else:
            # Essayer de trouver une colonne qui ressemble à un code département
            for col in gdf.columns:
                if 'code' in col.lower() or 'dept' in col.lower():
                    gdf['code'] = gdf[col].apply(normalize_department_code)
                    break
        
        # S'assurer qu'on a une colonne nom
        if 'nom' not in gdf.columns and 'NOM_DEPT' in gdf.columns:
            gdf['nom'] = gdf['NOM_DEPT']
        elif 'nom' not in gdf.columns:
            gdf['nom'] = gdf['code']
        
        return gdf
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du GeoJSON : {str(e)}")
        st.error("Assurez-vous que le fichier 'geoson.geojson' est présent dans le répertoire")
        return None

def aggregate_by_department(df):
    """Agrège les données par département"""
    agg_dict = {
        'Code du client': 'nunique',
        'Nb Visite': 'sum',
        'CA 2024': 'sum'
    }
    
    dept_stats = df.groupby('Departement').agg(agg_dict).reset_index()
    dept_stats.columns = ['Departement', 'Nb Clients', 'Nb Visite', 'CA 2024']
    
    return dept_stats

def get_centroids(gdf):
    """Calcule les centroïdes en reprojetant en EPSG:2154 puis reconvertit en EPSG:4326"""
    # Reprojeter en Lambert 93 (EPSG:2154)
    gdf_lambert = gdf.to_crs('EPSG:2154')
    
    # Calculer les centroïdes
    centroids = gdf_lambert.geometry.centroid
    
    # Reconvertir en WGS84 (EPSG:4326)
    centroids_wgs84 = centroids.to_crs('EPSG:4326')
    
    # Extraire lat/lon
    gdf['lat'] = centroids_wgs84.y
    gdf['lon'] = centroids_wgs84.x
    
    return gdf

def perform_clustering(gdf, n_zones):
    """Effectue le clustering sur les coordonnées géographiques"""
    # Préparer les données pour le clustering (seulement lat/lon)
    coords = gdf[['lat', 'lon']].values
    
    # Clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_zones,
        linkage='ward'
    )
    
    zone_labels = clustering.fit_predict(coords)
    
    # Convertir en noms de zones (Zone A, Zone B, etc.)
    zone_names = [f"Zone {string.ascii_uppercase[i]}" for i in zone_labels]
    
    gdf['Zone'] = zone_names
    
    return gdf

def create_folium_map(gdf):
    """Crée la carte Folium"""
    # Calculer le centre de la carte
    center_lat = gdf['lat'].mean()
    center_lon = gdf['lon'].mean()
    
    # Créer la carte
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Couleurs pour les zones
    unique_zones = sorted(gdf['Zone'].unique())
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    zone_colors = {zone: colors[i % len(colors)] for i, zone in enumerate(unique_zones)}
    
    # Ajouter les départements à la carte
    for idx, row in gdf.iterrows():
        # Créer le tooltip
        tooltip = f"Nom: {row.get('nom', 'N/A')}<br>Code: {row['code']}<br>Zone: {row['Zone']}"
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda feature, color=zone_colors[row['Zone']]: {
                'fillColor': color,
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6,
            },
            tooltip=tooltip
        ).add_to(m)
    
    return m

def create_indicators_table(gdf, dept_stats):
    """Crée le tableau d'indicateurs par zone"""
    # Merger les données géographiques avec les stats
    merged = gdf.merge(dept_stats, left_on='code', right_on='Departement', how='left')
    merged = merged.fillna(0)
    
    # Grouper par zone
    zone_indicators = []
    
    for zone in sorted(merged['Zone'].unique()):
        zone_data = merged[merged['Zone'] == zone]
        
        # Liste des départements
        dept_list = sorted(zone_data['code'].tolist())
        dept_str = ', '.join(dept_list)
        
        # Agrégations
        nb_clients = zone_data['Nb Clients'].sum()
        ca_2024 = zone_data['CA 2024'].sum()
        nb_visites = zone_data['Nb Visite'].sum()
        
        # ETP (diviseur constant 949)
        etp = nb_visites / 949
        
        # Ratios
        visites_per_etp = nb_visites / etp if etp > 0 else 0
        clients_per_etp = nb_clients / etp if etp > 0 else 0
        
        zone_indicators.append({
            'Zone': zone,
            'Départements': dept_str,
            'Nb Clients': int(nb_clients),
            'CA 2024 (€)': ca_2024,
            'Nb Visites': int(nb_visites),
            'ETP estimé': round(etp, 2),
            'Nb Visites / ETP': round(visites_per_etp, 1),
            'Nb Clients / ETP': round(clients_per_etp, 1)
        })
    
    return pd.DataFrame(zone_indicators)

# Interface utilisateur
st.title("🗺️ Sectorisation minimale (départements)")

# Sidebar
st.sidebar.header("Paramètres")
n_zones = st.sidebar.slider(
    "Nombre de zones",
    min_value=2,
    max_value=10,
    value=5
)

# Upload du fichier
uploaded_file = st.file_uploader(
    "Charger un fichier Excel (.xlsx)",
    type=['xlsx']
)

if uploaded_file is not None:
    # Charger et traiter les données
    df = load_and_process_data(uploaded_file)
    
    if df is not None:
        # Charger le GeoJSON
        gdf = load_geojson()
        
        if gdf is not None:
            # Agrégation par département
            dept_stats = aggregate_by_department(df)
            
            # Vérifier les départements manquants
            dept_in_data = set(dept_stats['Departement'].unique())
            dept_in_geo = set(gdf['code'].unique())
            
            missing_depts = dept_in_data - dept_in_geo
            if missing_depts:
                st.warning(f"Départements dans les données mais absents du GeoJSON : {sorted(list(missing_depts))}")
            
            # Filtrer les départements présents dans le GeoJSON
            dept_stats_filtered = dept_stats[dept_stats['Departement'].isin(dept_in_geo)]
            gdf_filtered = gdf[gdf['code'].isin(dept_in_data)].copy()
            
            if len(gdf_filtered) > 0:
                # Calculer les centroïdes
                gdf_filtered = get_centroids(gdf_filtered)
                
                # Clustering
                gdf_filtered = perform_clustering(gdf_filtered, n_zones)
                

                st.subheader("Carte des zones")
                # Créer et afficher la carte
                folium_map = create_folium_map(gdf_filtered)
                st_folium(folium_map, width=700, height=500)
                
                
                st.subheader("Indicateurs par zone")
                # Créer et afficher le tableau
                indicators_df = create_indicators_table(gdf_filtered, dept_stats_filtered)
                
                # Formater les colonnes numériques
                indicators_formatted = indicators_df.copy()
                indicators_formatted['CA 2024 (€)'] = indicators_formatted['CA 2024 (€)'].apply(
                    lambda x: f"{x:,.0f}".replace(',', ' ') if pd.notnull(x) else "0"
                )
                
                st.dataframe(
                    indicators_formatted,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error("Aucun département commun trouvé entre les données et le GeoJSON")
else:
    st.info("Veuillez charger un fichier Excel pour commencer l'analyse.")
    st.markdown("""
    **Format attendu du fichier Excel :**
    - `Code du client` : identifiant unique du magasin/client
    - `Departement` : code département (ex: "01", "2A", "2B", "75", "971"...)
    - `Nb Visite` : nombre de visites
    - `CA 2024` : chiffre d'affaires 2024
    
    **Note :** Assurez-vous que le fichier `geoson.geojson` des départements français est présent dans le répertoire.
    """)