import os
import tensorflow.keras.backend as K
import argparse

import pandas as pd
import numpy as np
from azureml.core import Run
import train_utils
import utils
import compile_model as cm
import time
from sklearn.utils import class_weight as cw
from tensorflow.keras.preprocessing.image import ImageDataGenerator

run = Run.get_context()

# Define source model name if fine tuning a model
SRC_MODEL_NAME = 'retcam_stage1'
DEST_MODEL_NAME = f'retcam_neo_2022_eff0---{utils.getStandardTime()}'

FIRST_EPOCHS = 1   
SECOND_EPOCHS = 10 
LR1 = 0.0088        #0.0088, 0.088, 0.00088, 0.02, 0.01
LR2 = 0.0001        #0.00037, 0.0037, 0.000037, 0.002,   0.00088, reduce by a factor of 10
DR1 = 0.235         # dropout rate 0.435
DR2 = 0.348         # 0.648
DENSE1 = 235
INPUT_SHAPE = (168, 224, 3)    #b0(168,224,3), b1(180,240,3)
BATCH_SIZE = 64
N_TRAIN_STEPS = 5

pos_label, neg_label = "1", "0"


model_parameters = {
    "LR1": LR1,
    "LR2": LR2,
    "DR1": DR1,
    "DR2": DR2,
    "DENSE1": DENSE1,
    "INPUT_SHAPE": INPUT_SHAPE
}

# Csvs names (combined retcam + neo)
train_csv_path = "retcam+neo_train_csv2022-04-13-11_00_00_validated.csv"
val_csv_path = "retcam+neo_val_csv2022-04-13-11_00_00_validated.csv"

# train_csv_path = 'neo_train_csv2022-04-07-15_00_00_validated.csv'
# val_csv_path = 'neo_val_csv2022-04-07-15_00_00_validated.csv'

srcPathCol = "fullPath"
targetCol = "hasROP"


def get_class_weight(labels):
    class_weight_arr = cw.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = {}
    for i in range(len(class_weight_arr)):
        class_weights[i] = class_weight_arr[i]
    return class_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, help="path to training data")
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument("--output_blob_path", type=str, help="path to output, directly to blob")
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs("cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    dir_path = args.dir_path
    output_blob_path = args.output_blob_path
    pre_train_weights_path = os.path.join(dir_path,"models","noisy.student.notop-b0.h5")

    print(f"Is pre_train_weights_path valid? : {os.path.exists(pre_train_weights_path)}")
    print(f"Pre_train_weights_path: {pre_train_weights_path}")
    
    print("dir_path", "=" * 50)
    print(dir_path)
    print("output_blob_path", "=" * 50)
    print(output_blob_path)
    
    # provide path to csvs for training 
    csvDir = os.path.join(dir_path, "binary_rop_classifier", "csvs")
    train_df = pd.read_csv(os.path.join(csvDir, train_csv_path))
    val_df = pd.read_csv(os.path.join(csvDir, val_csv_path))
    train_df = train_df[train_df["isExist"] == True]
    val_df = val_df[val_df["isExist"] == True]
    
    train_df[srcPathCol] = train_df[srcPathCol].apply(lambda x: os.path.join(dir_path, x))
    val_df[srcPathCol] = val_df[srcPathCol].apply(lambda x: os.path.join(dir_path, x))
    

    # Assigning labels
    train_df['hasROP'].loc[train_df['hasROP'] == True] = pos_label   #"Have ROP"
    train_df['hasROP'].loc[train_df['hasROP'] == False] = neg_label  #"No ROP"
    val_df['hasROP'].loc[val_df['hasROP'] == True] = pos_label       #"Have ROP" 
    val_df['hasROP'].loc[val_df['hasROP'] == False] = neg_label      #"No ROP"

    # sanity check if path is valid
    print("train_df: ")
    for idx,row in train_df.iterrows():
        if idx >= 10:
            break
        print(row[srcPathCol])
        print(os.path.exists(row[srcPathCol]))
    

    # Using ImagedataGenerators to form batches
    train_datagen = ImageDataGenerator(
        preprocessing_function=train_utils.preprocess_augment_fn   
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=train_utils.preprocess_fn
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col=srcPathCol,
        y_col=targetCol,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        validate_filenames=False
    )

    test_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col=srcPathCol,
        y_col=targetCol,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        validate_filenames=False
    )
    
print('hola amiguss')
print('pull pull pull request baby')
print('damn')


def subtract(a,b):
    
    
    print('hello world ')
    return a-b 
