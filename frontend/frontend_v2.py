import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)


import zipfile

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
import plotly.express as px


from src.config import DATA_DIR
from src.inference import fetch_next_hour_predictions, load_batch_of_features_from_store, fetch_hourly_rides, fetch_predictions
from src.plot_utils import plot_prediction

# Add parent directory to Python path


# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False


def visualize_predicted_demand(shapefile_path, predicted_demand):
    """
    Visualizes the predicted number of rides on a map of NYC taxi zones.

    Parameters:
        shapefile_path (str): Path to the NYC taxi zones shapefile.
        predicted_demand (dict): A dictionary where keys are taxi zone IDs (or names)
                                and values are the predicted number of rides.

    Returns:
        None
    """
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")

    # Ensure the taxi zone IDs in the shapefile match the keys in predicted_demand
    # Assuming the shapefile has a column 'zone_id' or 'LocationID' for taxi zones
    if "LocationID" not in gdf.columns:
        raise ValueError(
            "Shapefile must contain a 'LocationID' column to match taxi zones."
        )

    # Add a new column for predicted rides, defaulting to 0 if no prediction is available
    gdf["predicted_demand"] = gdf["LocationID"].map(predicted_demand).fillna(0)

    # Plot the map
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(
        column="predicted_demand",  # Column to color by
        cmap="OrRd",  # Color map (e.g., 'OrRd' for orange-red gradient)
        linewidth=0.8,
        ax=ax,
        edgecolor="black",
        legend=True,
        legend_kwds={"label": "Predicted Rides", "orientation": "vertical"},
    )

    # Add title and labels
    ax.set_title("Predicted NYC Taxi Rides by Zone", fontsize=16)
    ax.set_axis_off()  # Turn off axis for a cleaner map

    # Show the plot
    st.pyplot(fig)


def create_taxi_map(shapefile_path, prediction_data):
    """
    Create an interactive choropleth map of NYC taxi zones with predicted rides
    """
    # Load the NYC taxi zones shapefile
    nyc_zones = gpd.read_file(shapefile_path)

    # Merge with cleaned column names
    nyc_zones = nyc_zones.merge(
        prediction_data[["pickup_location_id", "predicted_demand"]],
        left_on="LocationID",
        right_on="pickup_location_id",
        how="left",
    )

    # Fill NaN values with 0 for predicted demand
    nyc_zones["predicted_demand"] = nyc_zones["predicted_demand"].fillna(0)

    # Convert to GeoJSON for Folium
    nyc_zones = nyc_zones.to_crs(epsg=4326)

    # Create map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")

    # Create color map
    colormap = LinearColormap(
        colors=[
            "#FFEDA0",
            "#FED976",
            "#FEB24C",
            "#FD8D3C",
            "#FC4E2A",
            "#E31A1C",
            "#BD0026",
        ],
        vmin=nyc_zones["predicted_demand"].min(),
        vmax=nyc_zones["predicted_demand"].max(),
    )

    colormap.add_to(m)

    # Define style function
    def style_function(feature):
        predicted_demand = feature["properties"].get("predicted_demand", 0)
        return {
            "fillColor": colormap(float(predicted_demand)),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    # Convert GeoDataFrame to GeoJSON
    zones_json = nyc_zones.to_json()

    # Add the choropleth layer
    folium.GeoJson(
        zones_json,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone", "predicted_demand"],
            aliases=["Zone:", "Predicted Demand:"],
            style=(
                "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
            ),
        ),
    ).add_to(m)

    # Store the map in session state
    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m


def load_shape_data_file(
    data_dir, url="https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", log=True
):
    """
    Downloads, extracts, and loads a shapefile as a GeoDataFrame.

    Parameters:
        data_dir (str or Path): Directory where the data will be stored.
        url (str): URL of the shapefile zip file.
        log (bool): Whether to log progress messages.

    Returns:
        GeoDataFrame: The loaded shapefile as a GeoDataFrame.
    """
    # Ensure data directory exists
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    zip_path = data_dir / "taxi_zones.zip"
    extract_path = data_dir / "taxi_zones"
    shapefile_path = extract_path / "taxi_zones.shp"

    # Download the file if it doesn't already exist
    if not zip_path.exists():
        if log:
            print(f"Downloading file from {url}...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            with open(zip_path, "wb") as f:
                f.write(response.content)
            if log:
                print(f"File downloaded and saved to {zip_path}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {url}: {e}")
    else:
        if log:
            print(f"File already exists at {zip_path}, skipping download.")

    # Extract the zip file if the shapefile doesn't already exist
    if not shapefile_path.exists():
        if log:
            print(f"Extracting files to {extract_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            if log:
                print(f"Files extracted to {extract_path}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Failed to extract zip file {zip_path}: {e}")
    else:
        if log:
            print(f"Shapefile already exists at {shapefile_path}, skipping extraction.")

    # Load and return the shapefile as a GeoDataFrame
    if log:
        print(f"Loading shapefile from {shapefile_path}...")
    try:
        gdf = gpd.read_file(shapefile_path).to_crs("epsg:4326")
        if log:
            print("Shapefile successfully loaded.")
        return gdf
    except Exception as e:
        raise Exception(f"Failed to load shapefile {shapefile_path}: {e}")


# st.set_page_config(layout="wide")

current_date = pd.Timestamp.now(tz="Etc/UTC")
st.title(f"New York Yellow Taxi Cab Demand Next Hour")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S")}')

progress_bar = st.sidebar.header("Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 4


with st.spinner(text="Download shape file for taxi zones"):
    geo_df = load_shape_data_file(DATA_DIR)
    st.sidebar.write("Shape file was downloaded")
    progress_bar.progress(1 / N_STEPS)


with st.spinner(text="Fetching batch of inference data"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write("Inference features fetched from the store")
    progress_bar.progress(2 / N_STEPS)


with st.spinner(text="Fetching predictions"):
    predictions = fetch_next_hour_predictions()
    st.sidebar.write("Model was loaded from the registry")
    progress_bar.progress(3 / N_STEPS)

shapefile_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"

with st.spinner(text="Plot predicted rides demand"):
    # predictions_df = visualize_predicted_demand(
    #     shapefile_path, predictions["predicted_demand"]
    # )
    st.subheader("Taxi Ride Predictions Map")
    map_obj = create_taxi_map(shapefile_path, predictions)

    # Display the map
    if st.session_state.map_created:
        st_folium(st.session_state.map_obj, width=800, height=600, returned_objects=[])

    # Display data statistics
    st.subheader("Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Rides",
            f"{predictions['predicted_demand'].mean():.0f}",
        )
    with col2:
        st.metric(
            "Maximum Rides",
            f"{predictions['predicted_demand'].max():.0f}",
        )
    with col3:
        st.metric(
            "Minimum Rides",
            f"{predictions['predicted_demand'].min():.0f}",
        )

    # Show sample of the data
    st.sidebar.write("Finished plotting taxi rides demand")
    progress_bar.progress(4 / N_STEPS)

st.dataframe(predictions.sort_values("predicted_demand", ascending=False).head(10))
top10 = (
    predictions.sort_values("predicted_demand", ascending=False)
    .head(10)["pickup_location_id"]
    .to_list()
)
for location_id in top10:
    fig = plot_prediction(
        features=features[features["pickup_location_id"] == location_id],
        prediction=predictions[predictions["pickup_location_id"] == location_id],
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
###########--------

location = {1: 'Newark Airport', 2: 'Jamaica Bay', 3: 'Allerton/Pelham Gardens', 4: 'Alphabet City', 5: 'Arden Heights', 6: 'Arrochar/Fort Wadsworth', 7: 'Astoria', 8: 'Astoria Park', 9: 'Auburndale', 10: 'Baisley Park', 11: 'Bath Beach', 12: 'Battery Park', 13: 'Battery Park City', 14: 'Bay Ridge', 15: 'Bay Terrace/Fort Totten', 16: 'Bayside', 17: 'Bedford', 18: 'Bedford Park', 19: 'Bellerose', 20: 'Belmont', 21: 'Bensonhurst East', 22: 'Bensonhurst West', 23: 'Bloomfield/Emerson Hill', 24: 'Bloomingdale', 25: 'Boerum Hill', 26: 'Borough Park', 27: 'Breezy Point/Fort Tilden/Riis Beach', 28: 'Briarwood/Jamaica Hills', 29: 'Brighton Beach', 30: 'Broad Channel', 31: 'Bronx Park', 32: 'Bronxdale', 33: 'Brooklyn Heights', 34: 'Brooklyn Navy Yard', 35: 'Brownsville', 36: 'Bushwick North', 37: 'Bushwick South', 38: 'Cambria Heights', 39: 'Canarsie', 40: 'Carroll Gardens', 41: 'Central Harlem', 42: 'Central Harlem North', 43: 'Central Park', 44: 'Charleston/Tottenville', 45: 'Chinatown', 46: 'City Island', 47: 'Claremont/Bathgate', 48: 'Clinton East', 49: 'Clinton Hill', 50: 'Clinton West', 51: 'Co-Op City', 52: 'Cobble Hill', 53: 'College Point', 54: 'Columbia Street', 55: 'Coney Island', 56: 'Corona', 57: 'Corona', 58: 'Country Club', 59: 'Crotona Park', 60: 'Crotona Park East', 61: 'Crown Heights North', 62: 'Crown Heights South', 63: 'Cypress Hills', 64: 'Douglaston', 65: 'Downtown Brooklyn/MetroTech', 66: 'DUMBO/Vinegar Hill', 67: 'Dyker Heights', 68: 'East Chelsea', 69: 'East Concourse/Concourse Village', 70: 'East Elmhurst', 71: 'East Flatbush/Farragut', 72: 'East Flatbush/Remsen Village', 73: 'East Flushing', 74: 'East Harlem North', 75: 'East Harlem South', 76: 'East New York', 77: 'East New York/Pennsylvania Avenue', 78: 'East Tremont', 79: 'East Village', 80: 'East Williamsburg', 81: 'Eastchester', 82: 'Elmhurst', 83: 'Elmhurst/Maspeth', 84: "Eltingville/Annadale/Prince's Bay", 85: 'Erasmus', 86: 'Far Rockaway', 87: 'Financial District North', 88: 'Financial District South', 89: 'Flatbush/Ditmas Park', 90: 'Flatiron', 91: 'Flatlands', 92: 'Flushing', 93: 'Flushing Meadows-Corona Park', 94: 'Fordham South', 95: 'Forest Hills', 96: 'Forest Park/Highland Park', 97: 'Fort Greene', 98: 'Fresh Meadows', 99: 'Freshkills Park', 100: 'Garment District', 101: 'Glen Oaks', 102: 'Glendale', 103: "Governor's Island/Ellis Island/Liberty Island", 104: "Governor's Island/Ellis Island/Liberty Island", 105: "Governor's Island/Ellis Island/Liberty Island", 106: 'Gowanus', 107: 'Gramercy', 108: 'Gravesend', 109: 'Great Kills', 110: 'Great Kills Park', 111: 'Green-Wood Cemetery', 112: 'Greenpoint', 113: 'Greenwich Village North', 114: 'Greenwich Village South', 115: 'Grymes Hill/Clifton', 116: 'Hamilton Heights', 117: 'Hammels/Arverne', 118: 'Heartland Village/Todt Hill', 119: 'Highbridge', 120: 'Highbridge Park', 121: 'Hillcrest/Pomonok', 122: 'Hollis', 123: 'Homecrest', 124: 'Howard Beach', 125: 'Hudson Sq', 126: 'Hunts Point', 127: 'Inwood', 128: 'Inwood Hill Park', 129: 'Jackson Heights', 130: 'Jamaica', 131: 'Jamaica Estates', 132: 'JFK Airport', 133: 'Kensington', 134: 'Kew Gardens', 135: 'Kew Gardens Hills', 136: 'Kingsbridge Heights', 137: 'Kips Bay', 138: 'LaGuardia Airport', 139: 'Laurelton', 140: 'Lenox Hill East', 141: 'Lenox Hill West', 142: 'Lincoln Square East', 143: 'Lincoln Square West', 144: 'Little Italy/NoLiTa', 145: 'Long Island City/Hunters Point', 146: 'Long Island City/Queens Plaza', 147: 'Longwood', 148: 'Lower East Side', 149: 'Madison', 150: 'Manhattan Beach', 151: 'Manhattan Valley', 152: 'Manhattanville', 153: 'Marble Hill', 154: 'Marine Park/Floyd Bennett Field', 155: 'Marine Park/Mill Basin', 156: 'Mariners Harbor', 157: 'Maspeth', 158: 'Meatpacking/West Village West', 159: 'Melrose South', 160: 'Middle Village', 161: 'Midtown Center', 162: 'Midtown East', 163: 'Midtown North', 164: 'Midtown South', 165: 'Midwood', 166: 'Morningside Heights', 167: 'Morrisania/Melrose', 168: 'Mott Haven/Port Morris', 169: 'Mount Hope', 170: 'Murray Hill', 171: 'Murray Hill-Queens', 172: 'New Dorp/Midland Beach', 173: 'North Corona', 174: 'Norwood', 175: 'Oakland Gardens', 176: 'Oakwood', 177: 'Ocean Hill', 178: 'Ocean Parkway South', 179: 'Old Astoria', 180: 'Ozone Park', 181: 'Park Slope', 182: 'Parkchester', 183: 'Pelham Bay', 184: 'Pelham Bay Park', 185: 'Pelham Parkway', 186: 'Penn Station/Madison Sq West', 187: 'Port Richmond', 188: 'Prospect-Lefferts Gardens', 189: 'Prospect Heights', 190: 'Prospect Park', 191: 'Queens Village', 192: 'Queensboro Hill', 193: 'Queensbridge/Ravenswood', 194: 'Randalls Island', 195: 'Red Hook', 196: 'Rego Park', 197: 'Richmond Hill', 198: 'Ridgewood', 199: 'Rikers Island', 200: 'Riverdale/North Riverdale/Fieldston', 201: 'Rockaway Park', 202: 'Roosevelt Island', 203: 'Rosedale', 204: 'Rossville/Woodrow', 205: 'Saint Albans', 206: 'Saint George/New Brighton', 207: 'Saint Michaels Cemetery/Woodside', 208: 'Schuylerville/Edgewater Park', 209: 'Seaport', 210: 'Sheepshead Bay', 211: 'SoHo', 212: 'Soundview/Bruckner', 213: 'Soundview/Castle Hill', 214: 'South Beach/Dongan Hills', 215: 'South Jamaica', 216: 'South Ozone Park', 217: 'South Williamsburg', 218: 'Springfield Gardens North', 219: 'Springfield Gardens South', 220: 'Spuyten Duyvil/Kingsbridge', 221: 'Stapleton', 222: 'Starrett City', 223: 'Steinway', 224: 'Stuy Town/Peter Cooper Village', 225: 'Stuyvesant Heights', 226: 'Sunnyside', 227: 'Sunset Park East', 228: 'Sunset Park West', 229: 'Sutton Place/Turtle Bay North', 230: 'Times Sq/Theatre District', 231: 'TriBeCa/Civic Center', 232: 'Two Bridges/Seward Park', 233: 'UN/Turtle Bay South', 234: 'Union Sq', 235: 'University Heights/Morris Heights', 236: 'Upper East Side North', 237: 'Upper East Side South', 238: 'Upper West Side North', 239: 'Upper West Side South', 240: 'Van Cortlandt Park', 241: 'Van Cortlandt Village', 242: 'Van Nest/Morris Park', 243: 'Washington Heights North', 244: 'Washington Heights South', 245: 'West Brighton', 246: 'West Chelsea/Hudson Yards', 247: 'West Concourse', 248: 'West Farms/Bronx River', 249: 'West Village', 250: 'Westchester Village/Unionport', 251: 'Westerleigh', 252: 'Whitestone', 253: 'Willets Point', 254: 'Williamsbridge/Olinville', 255: 'Williamsburg (North Side)', 256: 'Williamsburg (South Side)', 257: 'Windsor Terrace', 258: 'Woodhaven', 259: 'Woodlawn/Wakefield', 260: 'Woodside', 261: 'World Trade Center', 262: 'Yorkville East', 263: 'Yorkville West', 264: "nan", 265: 'Outside of NYC'}

# Get all unique location IDs from predictions
all_locations = predictions["pickup_location_id"].unique()

# Create a mapping of available IDs to names (fallback to ID if name is missing)
location_options = {loc_id: location.get(loc_id, f"Location ID {loc_id}") for loc_id in all_locations}

# Dropdown displaying location names instead of IDs
selected_location = st.selectbox(
    "Select a Location:",
    options=location_options.keys(),  # Use IDs internally
    format_func=lambda x: location_options[x]  # Show names in dropdown
)

########----------
# Ensure merged_df is created by merging ride data and predictions
merged_df = pd.merge(fetch_hourly_rides(24), fetch_predictions(24), on=["pickup_location_id", "pickup_hour"])

# Calculate the absolute error
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# Filter data for the selected location
filtered_df = merged_df[merged_df["pickup_location_id"] == selected_location]

# Check if data is available for the selected location
if filtered_df.empty:
    st.warning(f"No data available for {location_options[selected_location]} (ID: {selected_location})")
else:
    # Group by 'pickup_hour' and calculate the mean absolute error (MAE) for the selected location
    mae_selected_location = (
        filtered_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
    )
    mae_selected_location.rename(columns={"absolute_error": "MAE"}, inplace=True)

    # Create a Plotly plot for the selected location
    fig_selected = px.line(
        mae_selected_location,
        x="pickup_hour",
        y="MAE",
        title=f"Mean Absolute Error (MAE) for {location[selected_location]} (ID: {selected_location})",
        labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
        markers=True,
    )

    # Display the plot
    st.plotly_chart(fig_selected)
    
    # Display average MAE for the selected location
    st.write(f'**Average MAE for {location[selected_location]}:** {mae_selected_location["MAE"].mean():.2f}')