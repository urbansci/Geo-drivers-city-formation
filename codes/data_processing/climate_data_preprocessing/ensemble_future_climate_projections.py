"""
# -*- coding: utf-8 -*-
Author: Weiyu Zhang
Date: 2024
Description: CHELSA future climate ensemble mean calculation tool.
             Ensemble multiple climate model projection under each SSP scenarios.
             Applies proper scaling corrections and unit conversions according to CHELSA documentation.
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import logging
from datetime import datetime


class ClimateEnsembleProcessor:
    """
    Climate model ensemble processor for CHELSA future climate data.
    
    Computes ensemble means from multiple Global Climate Models (GCMs) under different
    SSP scenarios for bioclimatic variables. Handles data scaling, unit conversion,
    and statistical aggregation for robust climate projections.
    """
    
    def __init__(self, base_path=None):
        """
        Initialize the climate ensemble processor.
        
        Args:
            base_path (str): Base path to CHELSA future climate data
        """
        self.base_path = base_path or '/path/to/chelsa/future/climatologies/2071-2100'
        
        # Global Climate Models included in ensemble
        self.models = ['GFDL-ESM4', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'UKESM1-0-LL']
        
        # SSP scenarios to process
        self.scenarios = ['ssp126']  # Can be extended: ['ssp126', 'ssp370', 'ssp585']
        
        # Bioclimatic variables to process
        self.variables = ['bio1', 'bio4', 'bio7', 'bio8', 'bio9', 'bio12', 'bio15', 'bio18', 'bio19']
        
        # Temperature variables requiring Kelvin to Celsius conversion
        self.temperature_vars = ['bio1', 'bio8', 'bio9']
        
        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """
        Setup logging configuration for processing monitoring.
        """
        log_dir = os.path.join(self.base_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'ensemble_processing_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def get_file_path(self, model, scenario, variable):
        """
        Construct full file path for climate model data.
        
        Args:
            model (str): Climate model name
            scenario (str): SSP scenario name
            variable (str): Bioclimatic variable name
        Returns:
            str: Full path to the climate data file
        """
        filename = f'CHELSA_{variable}_2071-2100_{model.lower()}_{scenario}_V.2.1.tif'
        return os.path.join(self.base_path, model, scenario, 'bio', filename)

    def create_output_dir(self, scenario):
        """
        Create output directory for ensemble results.
        
        Args:
            scenario (str): SSP scenario name
        Returns:
            str: Path to output directory
        """
        output_dir = os.path.join(self.base_path, 'ensemble_mean', scenario)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def process_variable_data(self, data, variable):
        """
        Apply scaling and unit conversions to bioclimatic variable data.
        
        Args:
            data (numpy.ndarray): Raw variable data
            variable (str): Bioclimatic variable name
        Returns:
            numpy.ndarray: Processed data with proper scaling and units
        """
        # Ensure data is float32 for precise calculations
        data = data.astype(np.float32)
        
        # Apply variable-specific transformations
        if variable in self.temperature_vars:
            # Temperature variables: Convert from Kelvin*10 to Celsius
            return data * 0.1 - 273.15
        else:
            # Precipitation and other variables: Apply 0.1 scaling factor
            return data * 0.1

    def process_ensemble_mean(self, scenario, variable):
        """
        Process ensemble mean for a specific scenario and variable.
        
        Args:
            scenario (str): SSP scenario name
            variable (str): Bioclimatic variable name
        Returns:
            bool: Success status of processing
        """
        try:
            logging.info(f"Processing ensemble mean for {scenario} - {variable}")
            
            # Collect data from all climate models
            all_data = []
            metadata = None
            successful_models = []
            
            # Read data from each climate model
            for model in self.models:
                file_path = self.get_file_path(model, scenario, variable)
                
                try:
                    with rasterio.open(file_path) as src:
                        if metadata is None:
                            metadata = src.meta.copy()
                            # Update metadata to use float32
                            metadata.update(
                                dtype=rasterio.float32,
                                compress='lzw'
                            )
                        
                        # Read and process data
                        raw_data = src.read(1).astype(np.float32)
                        processed_data = self.process_variable_data(raw_data, variable)
                        all_data.append(processed_data)
                        successful_models.append(model)
                        
                        logging.info(f"Successfully processed model: {model}")
                        
                except Exception as e:
                    logging.warning(f"Failed to read {model} data for {variable}: {str(e)}")
                    continue
            
            # Check if we have sufficient data
            if len(all_data) == 0:
                raise ValueError(f"No valid data found for {scenario} - {variable}")
            
            if len(all_data) < len(self.models) / 2:
                logging.warning(f"Only {len(all_data)}/{len(self.models)} models available for {scenario} - {variable}")
            
            # Calculate ensemble mean
            logging.info(f"Calculating ensemble mean from {len(all_data)} models")
            data_array = np.array(all_data)
            logging.info(f"Data array shape: {data_array.shape}")
            
            ensemble_mean = np.mean(data_array, axis=0)
            
            # Save ensemble mean
            output_dir = self.create_output_dir(scenario)
            output_filename = f'ensemble_mean_{variable}.tif'
            output_path = os.path.join(output_dir, output_filename)
            
            with rasterio.open(output_path, 'w', **metadata) as dst:
                dst.write(ensemble_mean, 1)
            
            logging.info(f"Ensemble mean saved to: {output_path}")
            logging.info(f"Models included: {successful_models}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to process {scenario} - {variable}: {str(e)}")
            return False

    def process_all(self):
        """
        Process ensemble means for all scenarios and variables.
        """
        logging.info("Starting climate ensemble processing")
        logging.info(f"Base path: {self.base_path}")
        logging.info(f"Models: {self.models}")
        logging.info(f"Scenarios: {self.scenarios}")
        logging.info(f"Variables: {self.variables}")
        
        total_tasks = len(self.scenarios) * len(self.variables)
        completed_tasks = 0
        failed_tasks = 0
        
        # Process all scenario-variable combinations
        for scenario in self.scenarios:
            logging.info(f"Processing scenario: {scenario}")
            
            for variable in self.variables:
                success = self.process_ensemble_mean(scenario, variable)
                
                if success:
                    completed_tasks += 1
                else:
                    failed_tasks += 1
                
                # Progress reporting
                progress = (completed_tasks + failed_tasks) / total_tasks * 100
                logging.info(f"Progress: {progress:.1f}% ({completed_tasks + failed_tasks}/{total_tasks})")
        
        # Final summary
        logging.info("="*60)
        logging.info("ENSEMBLE PROCESSING SUMMARY")
        logging.info("="*60)
        logging.info(f"Total tasks: {total_tasks}")
        logging.info(f"Successful: {completed_tasks}")
        logging.info(f"Failed: {failed_tasks}")
        logging.info(f"Success rate: {completed_tasks/total_tasks*100:.1f}%")
        
        if completed_tasks == total_tasks:
            logging.info("All ensemble means calculated successfully!")
        else:
            logging.warning(f"Processing completed with {failed_tasks} failures")


def main():
    """
    Main function to execute climate ensemble processing.
    """
    # Initialize processor with default or custom base path
    base_path = './raw_dataset/CHELSA/future/chelsav2/GLOBAL/climatologies/2071-2100'
    processor = ClimateEnsembleProcessor(base_path=base_path)
    
    # Process all ensemble means
    processor.process_all()


if __name__ == "__main__":
    main()