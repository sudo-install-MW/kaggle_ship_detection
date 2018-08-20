import pandas as pd
import os

csv_path = '/home/mahesh/Universe/dataset/kaggle/train_ship_segmentations.csv'
img_path = '/home/mahesh/Universe/dataset/kaggle/train'

# Get image list
img_list = os.listdir(img_path)
print(len(img_list))

df = pd.read_csv(csv_path)
print(df[:100])
#print(df['ImageId'])