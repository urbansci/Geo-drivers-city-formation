"""
# -*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2024
Description: CHELSA climate data scaling correction and preprocessing tool.
             Raw CHELSA variables may suffer from magnitude offset issues due to GDAL version
             incompatibilities. This script implements scaling corrections according to official
             CHELSA documentation. Note that original data uses int8 format and must be converted
             to int32 or float format before rescaling to prevent overflow and precision loss.
"""


import os
import re
import rasterio
import numpy as np
from pathlib import Path


class ClimateDataProcessor:
    """
    CHELSA climate data processor for rescaling and preprocessing bioclimatic variables.
    
    Handles scaling factors and unit conversions for CHELSA v2.1 bioclimatic variables:
    - Temperature variables (bio1, bio8, bio9): Convert from Kelvin*10 to Celsius
    - Precipitation variables (bio4, bio7, bio12, bio15, bio18, bio19): Apply 0.1 scaling factor
    """
    
    def __init__(self, base_path=None, output_path=None):
        """
        Initialize the climate data processor.
        
        Args:
            base_path (str): Path to raw CHELSA data directory
            output_path (str): Path to output directory for processed data
        """
        self.base_path = base_path or '/path/to/chelsa/current/bio/'
        self.output_path = output_path or '/path/to/chelsa/processed/'
        
        # CHELSA bioclimatic variables to process
        self.variables = ['bio1', 'bio4', 'bio7', 'bio8', 'bio9', 'bio12', 'bio15', 'bio18', 'bio19']
        
        # Temperature variables that need Kelvin to Celsius conversion
        self.temperature_vars = ['bio1', 'bio8', 'bio9']
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
    def process_file(self, file_path, variable):
        """
        Process single CHELSA bioclimatic variable file.
        
        Args:
            file_path (str): Path to input raster file
            variable (str): Bioclimatic variable name (e.g., 'bio1')
        """
        try:
            with rasterio.open(file_path) as src:
                # Read data as integer (original CHELSA format)
                data = src.read(1).astype(np.float32)  # Convert to float32 for processing
                meta = src.meta.copy()
                
                # Update metadata for output format
                meta.update(
                    dtype=rasterio.float32,
                    compress='lzw'  # Add compression for efficient storage
                )
                
                # Apply variable-specific scaling and unit conversions
                if variable in self.temperature_vars:
                    # Temperature variables: Convert from Kelvin*10 to Celsius*10
                    # Formula: (value * 0.1) - 273.15, then scale back to integer
                    # This preserves precision while converting units
                    data = (data * 0.1 - 273.15)
                    print(f"Applied temperature conversion for {variable}")
                else:
                    # Precipitation and other variables: Apply 0.1 scaling factor
                    data = (data * 0.1)
                    print(f"Applied precipitation scaling for {variable}")
                
                # Generate output filename
                output_filename = f'CHELSA_{variable}_1981-2010_V.2.1_rescaled.tif'
                output_path = os.path.join(self.output_path, output_filename)
                
                # Save processed data
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(data, 1)
                    
                print(f"Successfully processed: {file_path}")
                print(f"Output saved to: {output_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    def process_all_files(self):
        """
        Process all CHELSA bioclimatic variable files in the base directory.
        """
        print("="*60)
        print("CHELSA CLIMATE DATA RESCALING")
        print("="*60)
        print(f"Input directory: {self.base_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Variables to process: {self.variables}")
        
        if not os.path.exists(self.base_path):
            print(f"ERROR: Input directory not found: {self.base_path}")
            return
        
        total_files = len(self.variables)
        processed_files = 0
        failed_files = []
        
        # Process each bioclimatic variable
        for variable in self.variables:
            # Construct input filename following CHELSA naming convention
            input_filename = f'CHELSA_{variable}_1981-2010_V.2.1.tif'
            input_file_path = os.path.join(self.base_path, input_filename)
            
            if os.path.exists(input_file_path):
                print(f"\nProcessing {variable} ({processed_files + 1}/{total_files})")
                self.process_file(input_file_path, variable)
                processed_files += 1
                
                # Calculate and display progress
                progress = (processed_files / total_files) * 100
                print(f"Progress: {progress:.1f}% ({processed_files}/{total_files})")
            else:
                print(f"WARNING: File not found: {input_file_path}")
                failed_files.append(variable)
        
        # Processing summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total variables: {total_files}")
        print(f"Successfully processed: {processed_files}")
        print(f"Failed/missing: {len(failed_files)}")
        
        if failed_files:
            print(f"Failed variables: {failed_files}")
        
        if processed_files == total_files:
            print("All files processed successfully!")
        else:
            print(f"Processing completed with {len(failed_files)} warnings.")


def main():
    """
    Main function to execute CHELSA data rescaling.
    """
    # Default paths - modify as needed
    input_path = './raw_dataset/CHELSA/current/chelsav2/GLOBAL/climatologies/1981-2010/bio'
    output_path = './raw_dataset/CHELSA/current/int/'
    
    # Initialize processor with custom paths
    processor = ClimateDataProcessor(base_path=input_path, output_path=output_path)
    
    # Process all files
    processor.process_all_files()


if __name__ == "__main__":
    main()