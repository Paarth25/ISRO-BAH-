import os
import glob
import numpy as np
import rasterio
import cv2

# 1. Input configuration path
INPUT_FOLDER = r"C:\Users\vegeta\Desktop\isro\original landsat data"

# 2. Setup the requested 5 structural dataset folders
OUTPUT_BASE_DIR = r"C:\Users\vegeta\Desktop\isro\data preprocessing\Separated_Bands"
band_folders = {
    "rgb_30m": os.path.join(OUTPUT_BASE_DIR, "B432_RGB_30m"),
    "tir_30m": os.path.join(OUTPUT_BASE_DIR, "B10_TIR_30m"),
    "rgb_100m": os.path.join(OUTPUT_BASE_DIR, "RGB_100m"),
    "tir_100m": os.path.join(OUTPUT_BASE_DIR, "TIR_100m"),
    "tir_200m": os.path.join(OUTPUT_BASE_DIR, "TIR_200m")
}

# Automatically instantiate directories
for folder in band_folders.values():
    os.makedirs(folder, exist_ok=True)

# 3. Locate all your multi-band combined scenes
all_scenes = glob.glob(os.path.join(INPUT_FOLDER, "L9_Global_Scene_*.tif"))
print(f"Found {len(all_scenes)} multi-band scenes to process.")

for scene_path in sorted(all_scenes):
    base_name = os.path.splitext(os.path.basename(scene_path))[0]
    print(f"\n--------------------------------------------------")
    print(f"Dynamically analyzing array context for: {base_name}")

    with rasterio.open(scene_path) as src:
        # Dynamically read exact pixel dimensions of the current file
        orig_height = src.height
        orig_width = src.width
        print(f"Detected Native Resolution Size: {orig_width} x {orig_height}")
        
        # Read the raw sensor bands from your stacked single source file
        # Order packed by Earth Engine script: 1:B2, 2:B3, 3:B4, 4:B10
        b2_data = src.read(1)
        b3_data = src.read(2)
        b4_data = src.read(3)
        b10_data = src.read(4)
        
        # Stack layers into a standard spatial format [Height, Width, Channels]
        # We merge B4(Red), B3(Green), B2(Blue)
        rgb_30m = np.stack([b4_data, b3_data, b2_data], axis=2)

        # -----------------------------------------------------------------
        # STEP 1: CALCULATE DYNAMIC DOWNSAMPLED GRID DIMENSIONS
        # -----------------------------------------------------------------
        # 100m downsample factor = 100 / 30 = 3.3333...
        w_100m = int(round(orig_width / (100.0 / 30.0)))
        h_100m = int(round(orig_height / (100.0 / 30.0)))
        
        # 200m downsample factor = 200 / 30 = 6.6666...
        w_200m = int(round(orig_width / (200.0 / 30.0)))
        h_200m = int(round(orig_height / (200.0 / 30.0)))
        
        print(f" -> Downsampling 100m targets to: {w_100m} x {h_100m}")
        print(f" -> Downsampling 200m targets to: {w_200m} x {h_200m}")

        # -----------------------------------------------------------------
        # STEP 2: RESIZE ARRAYS USING CV2 AREA INTERPOLATION
        # -----------------------------------------------------------------
        # Resize thermal infrared bands (B10)
        tir_100m = cv2.resize(b10_data, (w_100m, h_100m), interpolation=cv2.INTER_AREA)
        tir_200m = cv2.resize(b10_data, (w_200m, h_200m), interpolation=cv2.INTER_AREA)
        
        # Resize the aligned visible context layers (RGB) to 100m
        rgb_100m = cv2.resize(rgb_30m, (w_100m, h_100m), interpolation=cv2.INTER_AREA)

        # -----------------------------------------------------------------
        # STEP 3: WRITE GEOSPATIAL PRODUCTS TO CORRESPONDING FOLDERS
        # -----------------------------------------------------------------
        # Meta template configurations
        meta_30m = src.meta.copy()
        
        # 1. Save Original RGB 30m (3-Channel)
        meta_30m.update(count=3)
        out_rgb_30 = os.path.join(band_folders["rgb_30m"], f"{base_name}_RGB_30m.tif")
        with rasterio.open(out_rgb_30, 'w', **meta_30m) as dst:
            # Re-order to standard [Channels, Height, Width] rasterio format
            dst.write(rgb_30m[:,:,0], 1) # Red
            dst.write(rgb_30m[:,:,1], 2) # Green
            dst.write(rgb_30m[:,:,2], 3) # Blue

        # 2. Save Original TIR B10 30m (1-Channel)
        meta_30m.update(count=1)
        out_tir_30 = os.path.join(band_folders["tir_30m"], f"{base_name}_TIR_30m.tif")
        with rasterio.open(out_tir_30, 'w', **meta_30m) as dst:
            dst.write(b10_data, 1)

        # 3. Save Downsampled RGB 100m (3-Channel)
        # Calculate matching affine geotransform adjustments for 100m grid cell steps
        transform_100m = src.transform * src.transform.scale(
            (src.width / w_100m), (src.height / h_100m)
        )
        meta_100m = src.meta.copy()
        meta_100m.update(count=3, width=w_100m, height=h_100m, transform=transform_100m)
        out_rgb_100 = os.path.join(band_folders["rgb_100m"], f"{base_name}_RGB_100m.tif")
        with rasterio.open(out_rgb_100, 'w', **meta_100m) as dst:
            dst.write(rgb_100m[:,:,0], 1)
            dst.write(rgb_100m[:,:,1], 2)
            dst.write(rgb_100m[:,:,2], 3)

        # 4. Save Downsampled TIR 100m (1-Channel)
        meta_100m.update(count=1)
        out_tir_100 = os.path.join(band_folders["tir_100m"], f"{base_name}_TIR_100m.tif")
        with rasterio.open(out_tir_100, 'w', **meta_100m) as dst:
            dst.write(tir_100m, 1)

        # 5. Save Downsampled TIR 200m (1-Channel)
        transform_200m = src.transform * src.transform.scale(
            (src.width / w_200m), (src.height / h_200m)
        )
        meta_200m = src.meta.copy()
        meta_200m.update(count=1, width=w_200m, height=h_200m, transform=transform_200m)
        out_tir_200 = os.path.join(band_folders["tir_200m"], f"{base_name}_TIR_200m.tif")
        with rasterio.open(out_tir_200, 'w', **meta_200m) as dst:
            dst.write(tir_200m, 1)

        print(f" -> Successfully saved all structured resolutions for {base_name}")

print("\nAll tasks finalized! Check your folder layout paths.")
