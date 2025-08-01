# -*- coding: utf-8 -*-
"""
This script processes GAEZ v4 data to generate global potential agricultural calorie yield maps
for two historical periods: pre-1500 and post-1500.

The pipeline involves several key stages:
1.  Data Preprocessing: Clipping large-scale raster data into manageable grids.
2.  Mask Generation: Creating masks for specific agro-ecological zones and soil types (urban gaps).
3.  Spatial Interpolation: Filling in missing yield data in non-agricultural areas (e.g., cities)
    using Inverse Distance Weighting (IDW).
4.  Calorie Calculation: Converting crop yields (kg/ha) to calorie yields (kcal/ha).
5.  Max Yield Mapping: Determining the maximum potential calorie yield and the corresponding crop
    for each pixel on the globe, considering historical crop availability (pre- and post-1500).
6.  Visualization: Plotting the final result maps.
"""

# %%
# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import os
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.mask import mask as rasterio_mask
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# %%
# =============================================================================
# 2. DATA PREPROCESSING AND MASK GENERATION
# =============================================================================

def clip_resize(grids_path, raw_path, out_path, data_type):
    """
    Clips raw raster data to 1x1 degree grid cells and calculates the mean value for each cell.

    Args:
        grids_path (str): Path to the grid shapefile.
        raw_path (str): Path to the raw raster data to be clipped.
        out_path (str): Path to save the output file (the grid shapefile will be updated).
        data_type (str): The name for the new column in the GeoDataFrame to store the mean values.
    """
    # Read raster and grid data
    raster_data = rasterio.open(raw_path)
    grid_data = gpd.read_file(grids_path)
    grid_data[data_type] = -1000.0  # Initialize column

    # Loop through the grids
    desc = f"Clipping and Aggregating {os.path.basename(raw_path)}"
    for index, grid in tqdm(grid_data.iterrows(), total=grid_data.shape[0], desc=desc):
        # Create a MultiPolygon for masking
        multipolygon = MultiPolygon([grid['geometry']])

        # Clip the raster data using the grid cell geometry
        try:
            out_image, _ = rasterio_mask(raster_data, multipolygon.geoms, crop=True, all_touched=True)
            
            # Flatten and filter valid data points
            img_arr = out_image[0].flatten()
            img_arr = img_arr[img_arr >= -100] # Filter out no-data values
            
            # Calculate mean and store it
            if img_arr.size > 0:
                mean_val = np.mean(img_arr)
            else:
                mean_val = -100.0 # Assign no-data value if cell is empty

            grid_data.loc[index, data_type] = float(mean_val)
        except Exception as e:
            print(f"Could not process grid {grid['ID']}: {e}")
            grid_data.loc[index, data_type] = -100.0
            
    # Save the updated grid data
    grid_data.to_file(out_path)
    print(f"Updated grid file saved to {out_path}")


def generate_urban_gap_mask(hwsd_path, output_path, soil_id=32):
    """
    Generates a binary mask from the HWSD dominant soil type raster.
    This is used to identify urban areas or other gaps that need interpolation.

    Args:
        hwsd_path (str): Path to the HWSD dominant soil type GeoTIFF.
        output_path (str): Path to save the output binary mask GeoTIFF.
        soil_id (int): The soil ID to be masked (e.g., 32 might represent urban areas).
    """
    print("Generating urban gap mask...")
    with rasterio.open(hwsd_path) as hwsd:
        hwsd_data = hwsd.read(1)
        meta = hwsd.meta.copy()

        # Create binary mask: 1 for the target soil_id, 0 otherwise
        mask_data = np.zeros(hwsd_data.shape, dtype=np.float32)
        mask_data[hwsd_data == soil_id] = 1

        # Update metadata for the output file
        meta.update(dtype='float32', count=1)

        # Save the mask to a new GeoTIFF file
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask_data, 1)
    print(f"Urban gap mask saved to {output_path}")

# Example usage for mask generation:
# hwsd_domi_file = "/path/to/your/hwsd_domi.tif"
# close_gap_mask_file = "close_gap_mask.tif"
# generate_urban_gap_mask(hwsd_domi_file, close_gap_mask_file)


# %%
# =============================================================================
# 3. SPATIAL INTERPOLATION
# =============================================================================

def idw_interpolation(yield_raster_path, mask_raster_path, output_path, power=2, k_neighbors=5):
    """
    Fills gaps in a yield raster using Inverse Distance Weighting (IDW) interpolation.

    Args:
        yield_raster_path (str): Path to the potential yield raster with gaps.
        mask_raster_path (str): Path to the binary mask raster (1 for gaps, 0 for data).
        output_path (str): Path to save the interpolated raster.
        power (int): The power parameter for IDW.
        k_neighbors (int): The number of nearest neighbors to use for interpolation.
    """
    print(f"Performing IDW interpolation for {os.path.basename(yield_raster_path)}...")
    
    # Load data
    with rasterio.open(yield_raster_path) as src:
        potential_yield = src.read(1)
        meta = src.meta.copy()

    with rasterio.open(mask_raster_path) as src:
        mask = src.read(1)

    # Identify coordinates of known points (non-gaps) and target points (gaps)
    known_indices = np.where(mask == 0)
    target_indices = np.where(mask == 1)

    known_coords = np.array(list(zip(known_indices[0], known_indices[1])))
    known_values = potential_yield[known_indices]
    
    target_coords = list(zip(target_indices[0], target_indices[1]))

    # Build a k-d tree for efficient nearest neighbor search
    tree = cKDTree(known_coords)

    # Perform interpolation for each target point
    for y, x in tqdm(target_coords, desc="IDW Interpolating"):
        distances, locations = tree.query((y, x), k=k_neighbors)
        
        # Avoid division by zero if a target point is identical to a known point
        if np.any(distances < 1e-10):
            interpolated_value = known_values[locations[distances < 1e-10]][0]
        else:
            weights = 1.0 / np.power(distances, power)
            interpolated_value = np.sum(non_city_values[locations] * weights) / np.sum(weights)
        
        potential_yield[y, x] = interpolated_value

    # Save the interpolated result
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(potential_yield, 1)
    print(f"Interpolated raster saved to {output_path}")

def batch_interpolate_yields(raw_yield_dir, mask_path, output_dir):
    """
    Applies IDW interpolation to all crop yield rasters in a directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(raw_yield_dir):
        if filename.endswith(".tif"):
            raw_yield_path = os.path.join(raw_yield_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".tif", "_interpolated.tif"))
            
            if os.path.exists(output_path):
                print(f"Skipping {filename}, already interpolated.")
                continue
            
            idw_interpolation(raw_yield_path, mask_path, output_path)

# Example usage for interpolation:
# raw_yields_directory = "/home/zwy/Long-term-Global-Urban-Machine-Learning-Dataset/zwy/raw_dataset/GAEZ/Data used in QJE/"
# urban_mask_file = "close_gap_mask.tif"
# interpolated_output_directory = "/path/to/your/interpolated_yields/"
# batch_interpolate_yields(raw_yields_directory, urban_mask_file, interpolated_output_directory)


# %%
# =============================================================================
# 4. CALORIE YIELD CALCULATION
# =============================================================================

def convert_yield_to_calories(interpolated_dir, caloric_csv_path, output_dir):
    """
    Converts interpolated crop yields (kg/ha) to calorie yields (kcal/ha).
    """
    print("Converting yields to calories...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caloric_df = pd.read_csv(caloric_csv_path)

    for filename in os.listdir(interpolated_dir):
        if filename.endswith("_interpolated.tif"):
            # Extract crop acronym from filename
            try:
                crop_acronym = filename.split("_")[1]
            except IndexError:
                print(f"Could not parse crop acronym from {filename}. Skipping.")
                continue

            # Find caloric content for the crop
            if crop_acronym in caloric_df["Acronym"].values:
                caloric_content = caloric_df.loc[caloric_df["Acronym"] == crop_acronym, "Energy(kcal/g)"].values[0]
                # Note: yield (kg/ha) * caloric_content (kcal/g) -> yield (1000g/ha) * caloric_content (kcal/g)
                # The code seems to multiply directly, implying the yield unit might already be g/ha or
                # there is a factor of 1000 implicitly handled or ignored. Assuming direct multiplication is intended.
                
                input_path = os.path.join(interpolated_dir, filename)
                output_path = os.path.join(output_dir, filename.replace("_interpolated.tif", "_calory.tif"))

                with rasterio.open(input_path) as src:
                    potential_yield = src.read(1)
                    meta = src.meta.copy()

                    # Perform conversion
                    calorie_yield = potential_yield * caloric_content

                    # Save the result
                    with rasterio.open(output_path, 'w', **meta) as dst:
                        dst.write(calorie_yield, 1)
            else:
                print(f"Warning: Caloric content not found for crop '{crop_acronym}'. Skipping {filename}.")
    print("Calorie conversion complete.")

# Example usage:
# interpolated_dir = "/path/to/your/interpolated_yields/"
# caloric_table = "/path/to/your/caloric_contents.csv"
# calorie_results_dir = "/path/to/your/caloric_results/"
# convert_yield_to_calories(interpolated_dir, caloric_table, calorie_results_dir)


# %%
# =============================================================================
# 5. MAX YIELD MAPPING (PRE- & POST-1500)
# =============================================================================

def create_continent_mask(continents_shp_path, reference_raster_path, output_path):
    """
    Creates a raster mask where each pixel value is the ID of the continent it belongs to.
    """
    print("Creating continent mask...")
    continents_gdf = gpd.read_file(continents_shp_path)
    
    with rasterio.open(reference_raster_path) as ref_raster:
        meta = ref_raster.meta.copy()
        transform = ref_raster.transform
        out_shape = ref_raster.shape
    
    # Initialize mask with a no-data value
    continent_mask_array = np.full(out_shape, -1, dtype=np.int32)

    for index, continent in continents_gdf.iterrows():
        # Rasterize each continent polygon
        mask = geometry_mask([continent.geometry], invert=True, transform=transform, out_shape=out_shape)
        # Assign the continent's index (or a unique ID) to the pixels within the polygon
        continent_mask_array[mask] = index
    
    meta.update(dtype='int32', count=1, nodata=-1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(continent_mask_array, 1)
    print(f"Continent mask saved to {output_path}")


def generate_max_yield_map(calorie_yield_dir, continent_mask_path, crop_info_csv_path, output_prefix, period='post1500', crop_continent_csv_path=None):
    """
    Generates the final max calorie yield map and corresponding crop ID map.

    Args:
        calorie_yield_dir (str): Directory with calorie yield rasters.
        continent_mask_path (str): Path to the continent mask raster.
        crop_info_csv_path (str): Path to CSV with crop IDs and names.
        output_prefix (str): Prefix for the output files (e.g., 'max_yield_pre1500').
        period (str): 'pre1500' or 'post1500'. Determines which crops to consider.
        crop_continent_csv_path (str, optional): Path to CSV mapping pre-1500 crops to continents. Required if period is 'pre1500'.
    """
    print(f"Generating max yield map for period: {period}...")

    # Load crop ID mapping
    crop_id_df = pd.read_csv(crop_info_csv_path)
    crop_to_id = pd.Series(crop_id_df.Crop_ID.values, index=crop_id_df.Acronym).to_dict()

    # Load continent mask and metadata
    with rasterio.open(continent_mask_path) as mask_raster:
        continent_mask = mask_raster.read(1)
        meta = mask_raster.meta.copy()

    # Initialize arrays for max yield and crop ID
    max_yield = np.full(continent_mask.shape, -np.inf, dtype=np.float32)
    max_crop_id = np.full(continent_mask.shape, -1, dtype=np.int32) # -1 for no crop/sea

    # Get a list of all available crop acronyms from the directory
    available_crops = [f.split('_')[1] for f in os.listdir(calorie_yield_dir) if f.endswith("_calory.tif")]

    if period == 'pre1500':
        if not crop_continent_csv_path:
            raise ValueError("crop_continent_csv_path is required for 'pre1500' period.")
        crop_continent_df = pd.read_csv(crop_continent_csv_path)
        
        # Iterate through each continent
        for _, row in tqdm(crop_continent_df.iterrows(), total=len(crop_continent_df), desc="Processing Pre-1500"):
            continent_id = row['Continent_ID']
            native_crops_str = ','.join(row[['Cereals', 'Tubers', 'Pulses', 'Fruit']].astype(str))
            native_crops = [c.strip() for c in native_crops_str.replace(',', ' ').split() if c != '-']
            
            # Find the max yield among native crops for this continent
            for crop_acronym in native_crops:
                if crop_acronym in available_crops:
                    crop_id = crop_to_id.get(crop_acronym)
                    with rasterio.open(os.path.join(calorie_yield_dir, f"ylLr0_{crop_acronym}_calory.tif")) as src:
                        crop_yield = src.read(1)
                    
                    # Update max yield and crop ID only for pixels in the current continent
                    # and where the current crop's yield is higher
                    continent_pixels = (continent_mask == continent_id)
                    improvement_mask = (crop_yield > max_yield) & continent_pixels
                    
                    max_yield[improvement_mask] = crop_yield[improvement_mask]
                    max_crop_id[improvement_mask] = crop_id
    
    elif period == 'post1500':
        # Iterate through all available crops for all locations
        for crop_acronym in tqdm(available_crops, desc="Processing Post-1500"):
            crop_id = crop_to_id.get(crop_acronym)
            with rasterio.open(os.path.join(calorie_yield_dir, f"ylLr0_{crop_acronym}_calory.tif")) as src:
                crop_yield = src.read(1)
            
            # Update max yield and crop ID where the current crop's yield is higher
            improvement_mask = (crop_yield > max_yield)
            max_yield[improvement_mask] = crop_yield[improvement_mask]
            max_crop_id[improvement_mask] = crop_id

    # --- Finalize and save rasters ---
    # Handle no-data values
    sea_mask = (continent_mask == -1)
    no_yield_mask = (max_yield <= 0) | (max_yield == -np.inf)

    max_yield[no_yield_mask] = 0  # Land pixels with no yield potential
    max_yield[sea_mask] = -100 # Sea/No-data value for yield
    
    max_crop_id[no_yield_mask] = -2 # -2 for land with no yield
    max_crop_id[sea_mask] = -1      # -1 for sea/no-data

    # Save max yield raster
    yield_meta = meta.copy()
    yield_meta.update(dtype='float32', nodata=-100)
    with rasterio.open(f'{output_prefix}_yield.tif', 'w', **yield_meta) as dst:
        dst.write(max_yield, 1)
    
    # Save max crop ID raster
    crop_id_meta = meta.copy()
    crop_id_meta.update(dtype='int32', nodata=-1)
    with rasterio.open(f'{output_prefix}_crop_id.tif', 'w', **crop_id_meta) as dst:
        dst.write(max_crop_id, 1)

    print(f"Final maps saved with prefix: {output_prefix}")

# %%
# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def visualize_results(yield_path, crop_id_path, crop_info_csv_path, title_prefix):
    """
    Visualizes the final yield and crop ID maps.
    """
    # --- Visualize Max Yield ---
    with rasterio.open(yield_path) as src:
        yield_data = src.read(1)
        nodata = src.nodata
        yield_data[yield_data == nodata] = np.nan # Use NaN for plotting no-data

    plt.figure(figsize=(20, 10))
    plt.imshow(yield_data, cmap='viridis')
    plt.colorbar(label='Max Calorie Yield (kcal/ha)')
    plt.title(f'{title_prefix} - Maximum Potential Calorie Yield')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.show()

    # --- Visualize Max Crop ID ---
    crop_id_df = pd.read_csv(crop_info_csv_path)
    id_to_name = pd.Series(crop_id_df["Crop Name"].values, index=crop_id_df.Crop_ID).to_dict()
    # Add special values for legend
    id_to_name[-1] = 'Sea / No Data'
    id_to_name[-2] = 'Land / No Yield'

    with rasterio.open(crop_id_path) as src:
        crop_id_data = src.read(1)

    unique_ids = np.unique(crop_id_data)
    
    # Generate a colormap and legend patches
    cmap = plt.cm.get_cmap('tab20', len(unique_ids))
    patches = [mpatches.Patch(color=cmap(i), label=f"{val}: {id_to_name.get(val, 'Unknown')}")
               for i, val in enumerate(unique_ids)]

    fig, ax = plt.subplots(figsize=(20, 10))
    cax = ax.imshow(crop_id_data, cmap=cmap)
    ax.set_title(f'{title_prefix} - Dominant Crop for Max Yield')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    
    # Add legend
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

# =============================================================================
# SCRIPT EXECUTION (EXAMPLE)
# =============================================================================
if __name__ == '__main__':
    # --- IMPORTANT: Please adjust the paths below to match your local environment ---

    # Define the root directory where all your data is stored.
    # It's recommended to have separate 'input' and 'output' subdirectories.
    DATA_ROOT_DIR = './agricultur_data/'

    # --- Define INPUT Data Paths ---

    # Directory containing the raw crop yield GeoTIFFs (e.g., from GAEZ).
    # Each file should represent the potential yield for a single crop (e.g., 'ylLr0_whe.tif').
    RAW_YIELD_DIR = os.path.join(DATA_ROOT_DIR, 'input/raw_yields/')

    # Path to the dominant soil type GeoTIFF (e.g., from HWSD).
    # This is used to create a mask for non-agricultural areas (like cities) that need interpolation.
    URBAN_MASK_INPUT = os.path.join(DATA_ROOT_DIR, 'input/hwsd_domi.tif')

    # Path to the CSV file containing caloric information for each crop.
    # Must include columns: 'Acronym', 'Crop_ID', 'Crop Name', 'Energy(kcal/g)'.
    CALORIC_CSV = os.path.join(DATA_ROOT_DIR, 'input/tables/caloric_contents.csv')

    # Path to the world continents shapefile (.shp).
    # Used to create a raster mask that identifies which continent each pixel belongs to.
    CONTINENT_SHP = os.path.join(DATA_ROOT_DIR, 'input/continents/world-continents.shp')

    # Path to the CSV file that maps native crops to continents for the pre-1500 scenario.
    # Required only for generating the 'pre1500' map.
    CROP_CONTINENT_CSV = os.path.join(DATA_ROOT_DIR, 'input/tables/crop_continents.csv')

    # Path to a reference raster file (e.g., one of the raw yield files).
    # Its projection, resolution, and extent will be used as a template for creating new rasters.
    # NOTE: This is only a template for geospatial metadata (projection, etc.); the script processes ALL files in the directory, not just this one.
    REF_RASTER = os.path.join(RAW_YIELD_DIR, "ylLr0_whe.tif")

    # --- Define OUTPUT Data Paths ---
    # It's good practice to have a dedicated directory for all generated files.
    OUTPUT_DIR = os.path.join(DATA_ROOT_DIR, 'output/')

    # Path where the generated binary urban/gap mask will be saved.
    URBAN_MASK_OUTPUT = os.path.join(OUTPUT_DIR, "masks/close_gap_mask.tif")

    # Directory where the interpolated yield rasters will be saved.
    INTERPOLATED_DIR = os.path.join(OUTPUT_DIR, "interpolated_yields/")

    # Directory where the final calorie yield rasters will be saved.
    CALORIE_DIR = os.path.join(OUTPUT_DIR, "caloric_yields/")

    # Path where the generated continent ID raster mask will be saved.
    CONTINENT_MASK = os.path.join(OUTPUT_DIR, "masks/continents_mask.tif")
    
    # Create output directories if they don't exist to prevent errors.
    for path in [URBAN_MASK_OUTPUT, INTERPOLATED_DIR, CALORIE_DIR, CONTINENT_MASK]:
        os.makedirs(os.path.dirname(path), exist_ok=True)


    # --- Run Pipeline ---
    # The script is broken down into steps. Uncomment the steps you want to run.
    # Some steps are computationally intensive and only need to be run once.

    # Step 1: Generate a binary mask to identify gaps (e.g., urban areas) that need filling.
    print("--- Step 1: Generating Urban Gap Mask ---")
    generate_urban_gap_mask(URBAN_MASK_INPUT, URBAN_MASK_OUTPUT)

    # Step 2: Fill the gaps in all raw yield rasters using IDW interpolation.
    # This is a very time-consuming step.
    print("\n--- Step 2: Interpolating Raw Yields ---")
    batch_interpolate_yields(RAW_YIELD_DIR, URBAN_MASK_OUTPUT, INTERPOLATED_DIR)

    # Step 3: Convert the interpolated yields (kg/ha) into calorie yields (kcal/ha).
    print("\n--- Step 3: Converting Yields to Calories ---")
    convert_yield_to_calories(INTERPOLATED_DIR, CALORIC_CSV, CALORIE_DIR)

    # Step 4: Create a raster where each pixel's value corresponds to its continent's ID.
    print("\n--- Step 4: Creating Continent Mask ---")
    create_continent_mask(CONTINENT_SHP, REF_RASTER, CONTINENT_MASK)

    # Step 5: Generate the final max yield maps for both historical periods.
    print("\n--- Step 5: Generating Final Max Yield Maps ---")
    # For Pre-1500 (considers only native crops per continent)
    generate_max_yield_map(CALORIE_DIR, CONTINENT_MASK, CALORIC_CSV, os.path.join(OUTPUT_DIR, 'max_pre1500'), period='pre1500', crop_continent_csv_path=CROP_CONTINENT_CSV)
    # For Post-1500 (considers all crops globally)
    generate_max_yield_map(CALORIE_DIR, CONTINENT_MASK, CALORIC_CSV, os.path.join(OUTPUT_DIR, 'max_post1500'), period='post1500')
    
    # Step 6: Visualize the final output maps.
    print("\n--- Step 6: Visualizing Results ---")
    visualize_results(os.path.join(OUTPUT_DIR, 'max_pre1500_yield.tif'), os.path.join(OUTPUT_DIR, 'max_pre1500_crop_id.tif'), CALORIC_CSV, "Pre-1500")
    visualize_results(os.path.join(OUTPUT_DIR, 'max_post1500_yield.tif'), os.path.join(OUTPUT_DIR, 'max_post1500_crop_id.tif'), CALORIC_CSV, "Post-1500")

    print("\nScript finished.")