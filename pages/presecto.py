import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from folium.plugins import MarkerCluster, Fullscreen
import json

st.set_page_config(page_title="Analyse Sectorielle", layout="wide")

# Charger le fichier Excel
file_path = 'Calibrage France Direct Test (1).xlsx'  
magasins_data = pd.read_excel(file_path)
# Corriger les départements mal codés
magasins_data['Departement'] = magasins_data['Departement'].astype(str).str.zfill(2)

# Remplacer ou corriger manuellement si besoin
magasins_data['Departement'] = magasins_data['Departement'].replace({
    '20': '2A',  # ou '2B' selon la logique métier, ou dupliquer si nécessaire
    })

# Chemin local du fichier GeoJSON des départements français
LOCAL_GEOJSON_PATH = "geoson.geojson"

# Fonction pour charger les départements de France depuis le fichier GeoJSON local
@st.cache_data
def load_geojson_local():
    try:
        with open(LOCAL_GEOJSON_PATH, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Le fichier GeoJSON local est introuvable. Vérifiez le chemin.")
        return None
    
# Charger les données GeoJSON une seule fois
if "geojson_data" not in st.session_state:
    st.session_state["geojson_data"] = load_geojson_local()

geojson_data = st.session_state["geojson_data"]

# Définir les grandes zones (par départements, sans Île-de-France)
# Définir les grandes zones (par départements, sans Île-de-France)
zones = {
    "Nord": ['59', '62', '80', '76', '60'],
    "Nord-Ouest": ['22', '29', '35', '56', '50', '14', '61', '27', '76', '28', '72', '49', '44', '85','53','36','37','41','86','79','17','16'],
    "Nord-Est": ['08', '10', '21', '25', '39', '51', '52', '54', '55', '57', '67', '68', '70', '88', '90','45','18','89','58','71','02','01'],
    "Sud-Ouest": ['09', '11', '12', '19', '23', '24', '31', '32', '33', '40', '46', '47', '64', '65', '81', '82', '87','15','66','34','03','63'],
    "Sud-Est": ['03', '04', '05', '07', '13', '26', '30', '38', '42', '43', '48', '69', '73', '74', '83', '84', '2A', '2B', '06','98']
}
departements_ile_de_france = {
    '75': 'Paris',
    '77': 'Seine-et-Marne',
    '78': 'Yvelines',
    '91': 'Essonne',
    '92': 'Hauts-de-Seine',
    '93': 'Seine-Saint-Denis',
    '94': 'Val-de-Marne',
    '95': 'Val-d\'Oise'
}

# Associer une couleur à chaque grande zone
zone_colors = {
    "Nord": "#aaffa0",
    "Nord-Ouest": "#66ccff",
    "Nord-Est": "#ffff66",
    "Sud-Ouest": "#ff99ff",
    "Sud-Est": "#ffaa00",
}

# Liste des codes des départements de l'Île-de-France
ile_de_france_departments = ['75', '77', '78', '91', '92', '93', '94', '95']
# Initialisation des zones dans session_state (à faire AVANT d’utiliser get_zone_color)
if "zones_modifiables" not in st.session_state:
    st.session_state["zones_modifiables"] = zones.copy()

# Fonction pour déterminer la couleur d'un département en fonction de sa zone
def get_zone_color(department_code):
    if department_code in ile_de_france_departments:
        return "#ff6666"  # Couleur rouge pour l'Île-de-France
    for zone, departments in st.session_state["zones_modifiables"].items():
        if department_code in departments:
            return zone_colors[zone]
    return "#d9d9d9"  # Couleur par défaut pour les départements non classés

st.title("Sectorisation des départements français")

if "zones_modifiables" not in st.session_state:
    st.session_state["zones_modifiables"] = zones.copy()

zones_with_idf = st.session_state["zones_modifiables"].copy()
zones_with_idf["Île-de-France"] = ile_de_france_departments
# --- Filtre multisélection par zone ---
st.sidebar.markdown("### 🎯 Filtrer par zone")
zones_disponibles = list(zones_with_idf.keys())  # Toutes les zones
zones_selectionnees = st.sidebar.multiselect(
    "Zones à afficher :", zones_disponibles, default=zones_disponibles
)
# Ne garder que les zones sélectionnées
zones_with_idf = {z: d for z, d in zones_with_idf.items() if z in zones_selectionnees}
# # Récupérer tous les départements des zones sélectionnées
# selected_departments = sum(zones_with_idf.values(), [])

# # Filtrer les données de magasins en fonction des départements sélectionnés
# magasins_data_filtré = magasins_data[magasins_data['Departement'].astype(str).isin(selected_departments)]
# # Disposition des colonnes
# 🔁 Normaliser les formats des départements
magasins_data['Departement'] = magasins_data['Departement'].astype(str).str.zfill(2)
selected_departments = [str(dep).zfill(2) for dep in sum(zones_with_idf.values(), [])]

# 🐞 Affichage de vérification
# print("Départements dans le fichier :", magasins_data['Departement'].unique()[:10])
# print("Départements sélectionnés :", selected_departments[:10])

# 📦 Filtrage des données
magasins_data_filtré = magasins_data[magasins_data['Departement'].isin(selected_departments)]

# Vérifier quels départements sont absents du mapping
departements_fichier = set(magasins_data['Departement'].unique())
departements_zones = set(selected_departments)

departements_non_affectés = departements_fichier - departements_zones

print("Départements non pris en compte :", departements_non_affectés)
print("Nombre de lignes ignorées :", magasins_data[magasins_data['Departement'].isin(departements_non_affectés)].shape[0])
print(magasins_data['Departement'].unique())
# --- Sidebar : Option de modification du calcul ETP ---
st.sidebar.markdown("### 🔧 Options avancées")
modifier_etp = st.sidebar.checkbox("Modifier le calcul ETP")

# Valeur de référence pour ETP
if modifier_etp:
    diviseur_etp = st.sidebar.number_input("Valeur de référence pour le calcul ETP", value=949, step=1)
else:
    diviseur_etp = 949  # valeur par défaut
colA, colB = st.columns(2)
# Partie gauche (col1)
with colA:
    st.subheader("Indicateurs Clés")
    # 💡 Calcul des indicateurs
    nb_magasins_total = magasins_data_filtré["Code du client"].nunique()
    nb_visites_total = magasins_data_filtré["Nb Visite"].sum()
    ca_total_2023 = magasins_data_filtré["CA 2023"].sum()
    print("Avant filtrage :", len(magasins_data))
    print("Après filtrage :", len(magasins_data_filtré))
    print("Départements filtrés :", selected_departments)
    etp_total = round(nb_visites_total / diviseur_etp, 2)  # Utiliser la valeur de référence pour ETP

    # 💅 Style CSS pour les cards
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
    .card .green {
        color: green;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .delta {
        font-size: 1.2rem;
        margin-top: 5px;
    }
    .label {
        font-size: 1rem;
        color: #555;
    }
    .positive {
        color: green;
    }
    .negative {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)

    # ✅ Affichage des cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h2>{nb_magasins_total}</h2>
            <p>Magasins couverts</p>
            <div class="delta positive">100%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h2>{int(nb_visites_total)}</h2>
            <p>Nb total de Visites</p>
            <div class="delta positive">100%</div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h2>{ca_total_2023:,.0f} €</h2>
            <p>CA total</p>
            <div class="delta positive">100%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="card">
            <h2>{etp_total}</h2>
            <p>ETP estimés</p>
            <div class="delta positive">100%</div>
        </div>
        """, unsafe_allow_html=True)
            
    st.subheader("Carte géographique")
    if geojson_data:
        m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

        # Ajouter un bouton plein écran
        Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)

        # Ajouter les frontières des départements
        folium.GeoJson(
            geojson_data,
            name="Départements de France",
            style_function=lambda feature: {
                'fillColor': get_zone_color(feature['properties']['code']),
                'color': 'red' if feature['properties']['code'] in ile_de_france_departments else 'black',
                'weight': 3 if feature['properties']['code'] in ile_de_france_departments else 1,
                'fillOpacity': 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['nom', 'code'],
                aliases=['Nom du département: ', 'Numéro: '],
                localize=True
            )
        ).add_to(m)

        # Ajouter les magasins
        marker_cluster = MarkerCluster().add_to(m)

        for index, row in magasins_data_filtré.iterrows():
            icon_color = 'blue'
            icon_shape = 'info-sign'

            # Cas spécifique pour Monaco (département 98)
            if row['Departement'] == '98':
                icon_color = 'purple'
                icon_shape = 'star'

            folium.Marker(
                location=[row['lat'], row['long']],
                popup=f"{row['Nom du client']}<br>Adresse: {row['Adresse']}<br>CA 2023: {row['CA 2023']}",
                icon=folium.Icon(color=icon_color, icon=icon_shape)
            ).add_to(marker_cluster)



        # Afficher la carte dans l'application Streamlit
        st_folium(m, width=700, height=500, returned_objects=[])
        st.caption("Cette carte utilise des données GeoJSON des départements de France sectorisés.")

        # Calculer le nombre de magasins et le total du CA 2023 pour chaque département
        department_summary = magasins_data_filtré.groupby('Departement').agg(
            Nombre_Magasins=('Nom du client', 'count'),
            Total_CA_2023=('CA 2023', 'sum')
        ).reset_index()

        # Afficher le DataFrame dans Streamlit
        st.subheader("Nombre de magasins et CA total pour chaque département")
        st.dataframe(department_summary)

    
# Partie droite (col2)
with colB:
    # Ajouter un tableau pour résumer les données par zone
    codes_clients_déjà_vus = set()
    zone_summary = []

    for zone_name, department_list in zones_with_idf.items():
        magasins_in_zone = magasins_data_filtré[magasins_data_filtré['Departement'].astype(str).isin(department_list)]
        
        # Exclure les codes clients déjà comptés
        nouveaux_magasins = magasins_in_zone[~magasins_in_zone['Code du client'].isin(codes_clients_déjà_vus)]
        
        total_magasins = nouveaux_magasins['Code du client'].nunique()
        total_ca = nouveaux_magasins['CA 2023'].sum()
        total_visites = nouveaux_magasins['Nb Visite'].sum()
        etp = round(total_visites / diviseur_etp, 2)

        codes_clients_déjà_vus.update(nouveaux_magasins['Code du client'].unique())

        zone_summary.append({
            "Zone": zone_name,
            "Départements": ", ".join(department_list),
            "Nombre de Magasins": total_magasins,
            "Total CA (€)": f"{total_ca:,.2f}",
            "Nb Visites": total_visites,
            "ETP": etp
        })

    zone_summary_df = pd.DataFrame(zone_summary)

    st.subheader("Résumé des données par zone")
    st.dataframe(zone_summary_df, use_container_width=True)
    
    dupliqués = magasins_data.groupby('Code du client').size()
    st.write("Clients présents plusieurs fois :", (dupliqués > 1).sum())

    st.markdown("### 🛠 Modifier l'affectation d’un département")
    # Liste de tous les départements dans les zones (hors IDF)
    all_departments = sorted(set(dep for deps in st.session_state["zones_modifiables"].values() for dep in deps))

    # Sélection du département à déplacer
    departements_to_move = st.multiselect("Départements à déplacer :", all_departments)

    # Sélection de la nouvelle zone
    new_zone = st.selectbox("Nouvelle zone :", list(st.session_state["zones_modifiables"].keys()))

    # Bouton de mise à jour
    if st.button("Affecter les départements à la nouvelle zone"):
        moved = []
        for dep in departements_to_move:
            for z, deps in st.session_state["zones_modifiables"].items():
                if dep in deps:
                    st.session_state["zones_modifiables"][z].remove(dep)
                    break
            st.session_state["zones_modifiables"][new_zone].append(dep)
            moved.append(dep)
        if moved:
            st.success(f"✅ Les départements {', '.join(moved)} ont été déplacés vers la zone {new_zone}.")
            st.rerun()
        else:
            st.warning("Aucun département sélectionné.")



    # Option pour afficher chaque zone en détail
    for zone in zone_summary:
        with st.expander(f"Détails pour la zone {zone['Zone']}"):
            st.write(f"Départements : {zone['Départements']}")
            st.write(f"Nombre de Magasins : {zone['Nombre de Magasins']}")
            st.write(f"Total CA (€) : {zone['Total CA (€)']}")

    # Ajouter un expander pour afficher les départements de l'Île-de-France
    with st.expander("Voir les départements de zone rouge l'Île-de-France"):
        st.write("Voici la liste des départements de l'Île-de-France :")
        for code, nom in departements_ile_de_france.items():
            st.write(f"{code}: {nom}")

