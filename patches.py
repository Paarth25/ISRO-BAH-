import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window

# 1. Configuration Paths (Points to the output from your previous step)
BASE_DIR = r"C:\Users\vegeta\Desktop\isro\data preprocessing\Separated_Bands"
OUT_DIR = r"C:\Users\vegeta\Desktop\isro\ps10\dataset"
# Map out the exact input folders and corresponding configuration configurations
patch_tasks = [
    {
        "src_folder": os.path.join(BASE_DIR, "TIR_100m"),
        "out_folder": os.path.join(OUT_DIR, "TIR_100m_p256_s128"),
        "size": 256, "stride": 128, "pattern": "*_TIR_100m.tif"
    },
    {
        "src_folder": os.path.join(BASE_DIR, "TIR_100m"),
        "out_folder": os.path.join(OUT_DIR, "TIR_100m_p512_s256"),
        "size": 512, "stride": 256, "pattern": "*_TIR_100m.tif"
    },
    {
        "src_folder": os.path.join(BASE_DIR, "RGB_100m"),
        "out_folder": os.path.join(OUT_DIR, "RGB_100m_p256_s128"),
        "size": 256, "stride": 128, "pattern": "*_RGB_100m.tif"
    },
    {
        "src_folder": os.path.join(BASE_DIR, "TIR_200m"),
        "out_folder": os.path.join(OUT_DIR, "TIR_200m_p256_s128"),
        "size": 256, "stride": 128, "pattern": "*_TIR_200m.tif"
    }
]

# 2. Main Slicing Logic Execution Loop
for task in patch_tasks:
    os.makedirs(task["out_folder"], exist_ok=True)
    search_path = os.path.join(task["src_folder"], task["pattern"])
    target_files = glob.glob(search_path)
    
    print(f"\nProcessing Task folder: {os.path.basename(task['out_folder'])}")
    print(f" -> Found {len(target_files)} master images to slice.")
    
    p_size = task["size"]
    stride = task["stride"]
    
    for img_path in sorted(target_files):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        with rasterio.open(img_path) as src:
            height, width = src.height, src.width
            total_patches = 0
            
            # Step through width and height dimensions using explicit sliding stride loops
            for y in range(0, height - p_size + 1, stride):
                for x in range(0, width - p_size + 1, stride):
                    
                    # Instantiate spatial bounding box window
                    window = Window(x, y, p_size, p_size)
                    patch_data = src.read(window=window)
                    
                    # Filter out nodata black space regions to save storage space
                    if np.all(patch_data == 0) or np.any(np.isnan(patch_data)):
                        continue
                        
                    # Copy over geospatial headers and adjust resolution transform context
                    meta = src.meta.copy()
                    meta.update({
                        'height': p_size,
                        'width': p_size,
                        'transform': rasterio.windows.transform(window, src.transform)
                    })
                    
                    # Create a unique path file label sequence
                    out_filename = os.path.join(task["out_folder"], f"{base_name}_patch_y{y}_x{x}.tif")
                    with rasterio.open(out_filename, 'w', **meta) as dst:
                        dst.write(patch_data)
                        
                    total_patches += 1
                    
            print(f"   - {base_name}: Generated {total_patches} patches.")

print("\nAll 4 patched folder configurations created successfully!")
