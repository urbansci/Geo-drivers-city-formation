"""
Extract file information from a folder and save it as a parquet file.
This script is used to pre-load the images paths of training/inference datasets
Author: Junjie Yang
Date: 2025-01-10
"""
import os
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def parse_file(file_path):
    """
    Parse a file's path, extract Year and ID.
    :param file_path: full path of the file
    :return: (Year, ID, file_path)
    """
    file_name = os.path.basename(file_path)
    match = re.match(r"Y(\d{4})_(\d+)\.tiff$", file_name)
    if match:
        year = int(match.group(1))
        file_id = int(match.group(2))
        return year, file_id, file_path
    return None


def get_all_files(folder_path):
    """
    Get all file paths in the specified folder.
    :param folder_path: folder path to search for files
    :return: list of file paths
    """
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def process_files_in_batches(file_paths, num_workers=4):
    """
    Process file paths in parallel to extract Year and ID.
    :param file_paths: list of file paths to process
    :param num_workers: number of parallel workers
    :return: DataFrame containing Year, ID, file_path columns
    """
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # use tqdm to display progress bar
        for result in tqdm(executor.map(parse_file, file_paths), total=len(file_paths), desc="Processing files"):
            if result:  # filter out unmatched files
                results.append(result)
    # convert to DataFrame
    return pd.DataFrame(results, columns=["year", "id", "path"])


def main(folder_path, output_parquet, num_workers=4):
    """
    Extract file information and save as a parquet file.
    :param folder_path: folder path to search for files
    :param output_parquet: path of output filelist parquet file
    :param num_workers: number of parallel workers
    """
    print("Getting files' paths...")
    file_paths = get_all_files(folder_path)
    print(f"{len(file_paths)} files found")

    print("Start process files...")
    df = process_files_in_batches(file_paths, num_workers=num_workers)

    print("Save filelist as parquet file...")
    df.to_parquet(output_parquet, index=False)
    print(f"Filelist saved to: {output_parquet}")


if __name__ == "__main__":
    # Configure input and output paths

    folder_path = "/folder/path/to/search/for/files"  # folder containing the files to process
    output_parquet = "/path/of/output/filelist/parquet/filelist.parquet"  # output parquet file path

    num_workers = 4  # number of parallel workers

    main(folder_path, output_parquet, num_workers=num_workers)
