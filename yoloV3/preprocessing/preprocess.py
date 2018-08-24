from PIL import Image
import numpy as np
import pandas as pd

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )

def rle_to_mask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def read_csv_df():
    df = pd.read_csv('../Dataset/Ship_Detection/train_ship_segmentations.csv')
    return df

def preprocess_data(csv_df_in, output_csv_path):
    df_null = csv_df_in[pd.isnull(csv_df_in).any(axis=1)]
    csv_df = csv_df_in.dropna()
    height = 768
    width = 768
    output_df = pd.DataFrame(columns=['ImageId','BBox'])
    for index, row in csv_df.iterrows():
        #print(row['ImageId'], row['EncodedPixels'])
        imagemask = rle_to_mask(row['EncodedPixels'], height, width)
        first_nonzero_cols = first_nonzero(imagemask, axis=0, invalid_val=0)
        x_1 = first_nonzero(first_nonzero_cols, axis=0, invalid_val=-1)
        x_2 = last_nonzero(first_nonzero_cols, axis=0, invalid_val=-1)
        first_nonzero_rows = first_nonzero(imagemask, axis=1, invalid_val=0)
        y_1 = first_nonzero(first_nonzero_rows, axis=0, invalid_val=-1)
        y_2 = last_nonzero(first_nonzero_rows, axis=0, invalid_val=-1)
        # line = np.zeros(height, dtype=int)
        # line.fill(255)
        # imagemask[y_1, :] = line
        # imagemask[y_2, :] = line
        # imagemask[:, x_1] = line
        # imagemask[:, x_2] = line
        # save_image(imagemask, '../Dataset/1d3fdbd3a_mask.jpg')
        x = (x_1 + x_2) // 2
        y = (y_1 + y_2) // 2
        h = y_2 - y_1
        w = x_2 - x_1
        output_df.loc[len(output_df)] = [row['ImageId'], ''+str(x)+';' + str(y) + ';' + str(h) + ';' + str(w)]
    for index, row in df_null.iterrows():
        output_df.loc[len(output_df)] = [row['ImageId'], '-1;-1;-1;-1']

    output_df.to_csv(output_csv_path)

if __name__=="__main__":
    csv_df_in = read_csv_df()
    preprocess_data(csv_df_in, '../Dataset/Ship_Detection/train_ship_segmentations_bbox.csv')