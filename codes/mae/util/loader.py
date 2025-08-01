"""
Author: Weiyu Zhang, Junjie Yang
Date: 2025-02-09
Description: This script is used to provide custom dataset classes for loading and preprocessing DEM and climate data
"""
from PIL import ImageFilter
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
import pandas as pd
import tifffile


class ImageDatasetClimFromFilelist9Ch(Dataset):
    def __init__(self, filelist,types=None,transform=None):
        # Extract subfolders
        if types is None: 
            filelist = filelist
            self.data = pd.read_parquet(filelist)
            self.types = self.data.columns
            #self.data = self.data.reset_index(drop=True)
        
        self.transform = transform
        if self.transform is None:
            augmentation = []
            self.transform = transforms.Compose(augmentation)
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        ave_temp = Image.open(self.data.at[idx, "bio_1"]).convert('F')
        std_temp = Image.open(self.data.at[idx, "bio_4"]).convert('F')
        range_temp = Image.open(self.data.at[idx, "bio_7"]).convert('F')
        temp_wet = Image.open(self.data.at[idx, "bio_8"]).convert('F')
        temp_dry = Image.open(self.data.at[idx, "bio_9"]).convert('F')
        ave_prec = Image.open(self.data.at[idx, "bio_12"]).convert('F')
        std_prec = Image.open(self.data.at[idx, "bio_15"]).convert('F')
        prec_hot = Image.open(self.data.at[idx, "bio_18"]).convert('F')
        prec_cold = Image.open(self.data.at[idx, "bio_19"]).convert('F')
        
        combined_image = torch.stack(
            [
            transforms.ToTensor()(ave_temp)[0],
            transforms.ToTensor()(std_temp)[0],
            transforms.ToTensor()(range_temp)[0],
            transforms.ToTensor()(temp_wet)[0],
            transforms.ToTensor()(temp_dry)[0],
            transforms.ToTensor()(ave_prec)[0],
            transforms.ToTensor()(std_prec)[0],
            transforms.ToTensor()(prec_hot)[0],
            transforms.ToTensor()(prec_cold)[0]
            ])
        
        # check for nan
        if torch.isnan(combined_image).any():
            print('nan found in image:', self.data.at[idx, "bio_1"])
        
        # Apply transformations
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        return combined_image

class ImageDatasetClimAll(Dataset):
    def __init__(self, folder,types=None,transform=None):
        # Extract subfolders
        if types is None: 
            self.folder = folder
            self.types = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
            self.folders = []
        for i in range(len(self.types)):
            if "bio" in self.types[i]:
                if int(self.types[i].split('_')[1]) == 1:
                    self.ave_temp_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 4:
                    self.std_temp_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 7:
                    self.range_temp_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 8:
                    self.temp_wet_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 9:
                    self.temp_dry_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 12:
                    self.ave_prec_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 15:
                    self.std_prec_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 18:
                    self.prec_hot_folder = os.path.join(self.folder, self.types[i])
                elif int(self.types[i].split('_')[1]) == 19:
                    self.prec_cold_folder = os.path.join(self.folder, self.types[i])
                else:        
                    raise ValueError('Invalid type: {}'.format(self.types[i]))
            else:
                print('Invalid type: {}'.format(self.types[i]))
                continue
                #raise ValueError('Invalid type: {}'.format(self.types[i]))
        self.files = [f for f in os.listdir(self.ave_temp_folder) if os.path.isfile(os.path.join(self.ave_temp_folder, f))]

        self.transform = transform
        if self.transform is None:
            augmentation = []
            self.transform = transforms.Compose(augmentation)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        ave_temp = Image.open(os.path.join(self.ave_temp_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        std_temp = Image.open(os.path.join(self.std_temp_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        range_temp = Image.open(os.path.join(self.range_temp_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        temp_wet = Image.open(os.path.join(self.temp_wet_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        temp_dry = Image.open(os.path.join(self.temp_dry_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        ave_prec = Image.open(os.path.join(self.ave_prec_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        std_prec = Image.open(os.path.join(self.std_prec_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        prec_hot = Image.open(os.path.join(self.prec_hot_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        prec_cold = Image.open(os.path.join(self.prec_cold_folder, self.files[idx])).convert('F')  # Convert to grayscale for consistency
        
        combined_image = torch.stack(
            [
            torch.tensor(np.array(ave_temp)).float(), 
            torch.tensor(np.array(std_temp)).float(), 
            torch.tensor(np.array(range_temp)).float(),
            torch.tensor(np.array(temp_wet)).float(),
            torch.tensor(np.array(temp_dry)).float(),
            torch.tensor(np.array(ave_prec)).float(), 
            torch.tensor(np.array(std_prec)).float(),
            torch.tensor(np.array(prec_hot)).float(),
            torch.tensor(np.array(prec_cold)).float()
            ])

        # Apply transformations
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        return combined_image

class ImageDatasetWithBinaryWater(Dataset):
    def __init__(self, folder,types=None,transform=None):
        # Extract subfolders
        if types is None: 
            self.folder = folder
            self.subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
            self.types = self.subfolders
        else:
            self.types = ['dem']
        
        for i in range(len(self.types)):
            if self.types[i] == 'dem':
                self.dem_folder = os.path.join(self.folder, self.types[i])
            elif self.types[i] == 'water':
                self.water_folder = os.path.join(self.folder, self.types[i])
            else:
                print('Invalid type: {}'.format(self.types[i]))
                continue
                #raise ValueError('Invalid type: {}'.format(self.types[i]))
        self.dem_files = [f for f in os.listdir(self.dem_folder) if os.path.isfile(os.path.join(self.dem_folder, f))]
        self.water_files = [f for f in os.listdir(self.water_folder) if os.path.isfile(os.path.join(self.water_folder, f))]

        self.transform = transform
        if self.transform is None:
            augmentation = [transforms.ToTensor()]
            self.transform = transforms.Compose(augmentation)

        # 将数据压缩到文件
        if not os.path.exists(os.path.join(self.folder,"dem_file_list.pkl")):
            print("============>save dem_file_list.pkl", os.path.join(self.folder,"dem_file_list.pkl"))
            with open(os.path.join(self.folder,"dem_file_list.pkl"), 'wb') as file:
                pickle.dump(self.dem_files, file)
            print("============>save finished")
    def __len__(self):
        return len(self.dem_files)

    def __getitem__(self, idx):
        dem_path = os.path.join(self.dem_folder, self.dem_files[idx])
        water_path = os.path.join(self.water_folder, self.water_files[idx])

        dem_image = Image.open(dem_path).convert('F')  # Convert to grayscale for consistency
        water_image = Image.open(water_path)  # Convert to grayscale

        # Ensure water image is binary
        water_image = torch.tensor(np.array(water_image)).float()
        water_image = torch.where(water_image > 0, 1, 0)

        # Combine images into a 2-channel image
        combined_image = torch.stack([torch.tensor(np.array(dem_image)).float(), water_image])

        # Apply transformations
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        # Normalize DEM channel
        combined_image[0] = (combined_image[0] - combined_image[0].mean()) / (combined_image[0].std()+1e-6) # avoid nan
        # Convert water to -1 and 1
        #combined_image[1] = torch.where(combined_image[1] > 0.25, 1, -1)
        combined_image[1] = combined_image[1] * 2 - 1

        return combined_image

class ImageValDatasetWithBinaryWater(Dataset):
    def __init__(self, folder,types=None,transform=None):
        # Extract subfolders
        if types is None: 
            self.folder = folder
            print(folder)
            # print(os.listdir(folder))
            self.subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
            self.types = self.subfolders
        else:
            self.types = ['dem']
        
        for i in range(len(self.types)):
            if self.types[i] == 'dem':
                self.dem_folder = os.path.join(self.folder, self.types[i])
            elif self.types[i] == 'water':
                self.water_folder = os.path.join(self.folder, self.types[i])
            else:
                print('Invalid type: {}'.format(self.types[i]))
                continue
                #raise ValueError('Invalid type: {}'.format(self.types[i]))
        self.dem_files = [f for f in os.listdir(self.dem_folder) if os.path.isfile(os.path.join(self.dem_folder, f))]
        self.water_files = [f for f in os.listdir(self.water_folder) if os.path.isfile(os.path.join(self.water_folder, f))]

    def __len__(self):
        return len(self.dem_files)

    def __getitem__(self, idx):
        dem_path = os.path.join(self.dem_folder, self.dem_files[idx])
        water_path = os.path.join(self.water_folder, self.water_files[idx])

        dem_image = torch.tensor(np.array(Image.open(dem_path).convert('F'))).float()  # Convert to grayscale for consistency
        water_image = Image.open(water_path).convert('F')  # Convert to grayscale

        # Ensure water image is binary
        water_image = torch.tensor(np.array(water_image)).float()
        water_image = torch.where(water_image > 0, 1, 0)

        # Combine images into a 2-channel image
        combined_image = torch.stack([dem_image, water_image])

        mean_dem = combined_image[0].mean()
        std_dem = combined_image[0].std()
        # Normalize DEM channel
        combined_image[0] = (combined_image[0] - combined_image[0].mean()) / (combined_image[0].std()+1e-6) # avoid nan
        # Convert water to -1 and 1
        combined_image[1] = combined_image[1] * 2 - 1
        
        
        return combined_image, dem_image.mean(), dem_image.std(), self.dem_files[idx][:-5].split('_')[1]

# 2025.1.13 by YJJ, for DEM & water parquet filelist
class ImageValDatasetWithBinaryWaterParquetFilelist(Dataset):
    def __init__(self, filelist, types=None, transform=None):
        self.data = pd.read_parquet(filelist)
        # 不涉及transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        dem_path = self.data.at[idx, 'dem_path']
        water_path = self.data.at[idx, 'water_path']
        
        dem_image = torch.tensor(np.array(Image.open(dem_path).convert('F'))).float()  # Convert to grayscale for consistency
        water_image = Image.open(water_path).convert('F')  # Convert to grayscale

        # Ensure water image is binary
        water_image = torch.tensor(np.array(water_image)).float()
        water_image = torch.where(water_image > 0, 1, 0)

        # Combine images into a 2-channel image
        combined_image = torch.stack([dem_image, water_image])

        # Normalize DEM channel
        combined_image[0] = (combined_image[0] - combined_image[0].mean()) / (combined_image[0].std()+1e-6) # avoid nan
        # Convert water to -1 and 1
        combined_image[1] = combined_image[1] * 2 - 1
        
        return combined_image, dem_image.mean(), dem_image.std(), self.data.at[idx, 'id']

class ImageValDatasetClimAll(Dataset):
    def __init__(self, filelist,year = 20, types=None,transform=None):
        # Extract subfolders
        if types is None: 
            filelist = filelist
            self.data = pd.read_parquet(filelist)
            self.data = self.data[self.data['year'] == year]
            self.data = self.data.reset_index(drop=True)

        self.transform = transform
        if self.transform is None:
            augmentation = [
                transforms.Normalize(
                    mean=[8.757, 806.940, 31.380, 16.095, 3.403, 717.817, 63.785, 201.198,142.541 ], 
                    std=[14.718, 500.542, 13.907, 9.944, 20.500, 726.910, 37.967, 199.588, 237.565])
            ]
            self.transform = transforms.Compose(augmentation)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):

        ave_temp = Image.open(self.data.at[idx, "bio_1"]).convert('F')
        std_temp = Image.open(self.data.at[idx, "bio_4"]).convert('F')
        range_temp = Image.open(self.data.at[idx, "bio_7"]).convert('F')
        temp_wet = Image.open(self.data.at[idx, "bio_8"]).convert('F')
        temp_dry = Image.open(self.data.at[idx, "bio_9"]).convert('F')
        ave_prec = Image.open(self.data.at[idx, "bio_12"]).convert('F')
        std_prec = Image.open(self.data.at[idx, "bio_15"]).convert('F')
        prec_hot = Image.open(self.data.at[idx, "bio_18"]).convert('F')
        prec_cold = Image.open(self.data.at[idx, "bio_19"]).convert('F')
        
        combined_image = torch.stack(
            [
            transforms.ToTensor()(ave_temp)[0],
            transforms.ToTensor()(std_temp)[0],
            transforms.ToTensor()(range_temp)[0],
            transforms.ToTensor()(temp_wet)[0],
            transforms.ToTensor()(temp_dry)[0],
            transforms.ToTensor()(ave_prec)[0],
            transforms.ToTensor()(std_prec)[0],
            transforms.ToTensor()(prec_hot)[0],
            transforms.ToTensor()(prec_cold)[0]
            ])
        
        # Apply transformations
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        return combined_image, -1, -1, self.data.at[idx, "bio_1"].split('_')[-1][:-5].split('.')[0]

# 2025.1.9 by YJJ, for multiband-climate-dataset
class ImageValDatasetClimAllMultiBand(Dataset):
    def __init__(self, filelist,year = 20, types=None,transform=None):
        # Extract subfolders
        if types is None: 
            filelist = filelist
            self.data = pd.read_parquet(filelist)
            self.data = self.data[self.data['year'] == year]
            self.data = self.data.reset_index(drop=True)
            #self.data = self.data.reset_index(drop=True)
    
        #self.files = [f for f in os.listdir(self.ave_temp_folder) if os.path.isfile(os.path.join(self.ave_temp_folder, f))]

        self.transform = transform
        if self.transform is None:
            augmentation = [
                transforms.Normalize(
                    mean=[8.757, 806.940, 31.380, 16.095, 3.403, 717.817, 63.785, 201.198,142.541 ], 
                    std=[14.718, 500.542, 13.907, 9.944, 20.500, 726.910, 37.967, 199.588, 237.565])
            ]
            self.transform = transforms.Compose(augmentation)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):

        multi_band_image = tifffile.imread(self.data.at[idx, "path"])
        # ave_temp = Image.open(self.data.at[idx, "bio_1"]).convert('F')
        bands = []
        for band_idx in range(multi_band_image.shape[2]):
            temp = multi_band_image[:,:,band_idx]
            band_tensor = transforms.ToTensor()(temp)[0]
            bands.append(band_tensor)
        combined_image = torch.stack(bands)
        
        # Apply transformations
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        # return combined_image, -1, -1, self.data.at[idx, "bio_1"].split('_')[-1][:-5].split('.')[0]
        return combined_image, -1, -1, self.data.at[idx, "id"]

# 2025.2.6 by YJJ, Revise version for multiband-climate-dataset
# Read the multi-band file with the right order
# right order (transform parameter order): [1, 4, 7, 8, 9, 12, 15, 18, 19]
class ImageValDatasetClimAllMultiBandRevise(Dataset):
    def __init__(self, filelist,year = 20, types=None,transform=None):
        # Extract subfolders
        if types is None: 
            filelist = filelist
            self.data = pd.read_parquet(filelist)
            self.data = self.data[self.data['year'] == year]
            self.data = self.data.reset_index(drop=True)
            #self.data = self.data.reset_index(drop=True)
    
        #self.files = [f for f in os.listdir(self.ave_temp_folder) if os.path.isfile(os.path.join(self.ave_temp_folder, f))]

        self.transform = transform
        if self.transform is None:
            augmentation = [
                transforms.Normalize(
                    mean=[8.757, 806.940, 717.817, 63.785, 31.380, 16.095, 3.403, 201.198, 142.541], 
                    std=[14.718, 500.542, 726.910, 37.967, 13.907, 9.944, 20.500, 199.588, 237.565]
                    )
            ]
            self.transform = transforms.Compose(augmentation)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):

        multi_band_image = tifffile.imread(self.data.at[idx, "path"])
        # ave_temp = Image.open(self.data.at[idx, "bio_1"]).convert('F')
        bands = []
        for band_idx in range(multi_band_image.shape[2]):
            temp = multi_band_image[:,:,band_idx]
            band_tensor = transforms.ToTensor()(temp)[0]
            bands.append(band_tensor)
        combined_image = torch.stack(bands)
        
        # Apply transformations
        if self.transform is not None:
            combined_image = self.transform(combined_image)

        # return combined_image, -1, -1, self.data.at[idx, "bio_1"].split('_')[-1][:-5].split('.')[0]
        return combined_image, -1, -1, self.data.at[idx, "id"]
        # 最后一个返回值为ID

