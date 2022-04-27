import os
import argparse


# Define source model name if fine tuning a model
SRC_MODEL_NAME = "retcam_stage1"

FIRST_EPOCHS = 1
SECOND_EPOCHS = 10
LR1 = 0.0088  # 0.0088, 0.088, 0.00088, 0.02, 0.01
LR2 = 0.0001  # 0.00037, 0.0037, 0.000037, 0.002,   0.00088, reduce by a factor of 10
DR1 = 0.235  # dropout rate 0.435
DR2 = 0.348  # 0.648
DENSE1 = 235
INPUT_SHAPE = (168, 224, 3)  # b0(168,224,3), b1(180,240,3)
BATCH_SIZE = 64
N_TRAIN_STEPS = 5

pos_label, neg_label = "1", "0"


model_parameters = {
    "LR1": LR1,
    "LR2": LR2,
    "DR1": DR1,
    "DR2": DR2,
    "DENSE1": DENSE1,
    "INPUT_SHAPE": INPUT_SHAPE,
}

# Csvs names (combined retcam + neo)
train_csv_path = "retcam+neo_train_csv2022-04-13-11_00_00_validated.csv"
val_csv_path = "retcam+neo_val_csv2022-04-13-11_00_00_validated.csv"

# train_csv_path = 'neo_train_csv2022-04-07-15_00_00_validated.csv'
# val_csv_path = 'neo_val_csv2022-04-07-15_00_00_validated.csv'

srcPathCol = "fullPath"
targetCol = "hasROP"


def get_class_weight(labels):
    class_weights = {}
    return class_weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, help="path to training data")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument(
        "--output_blob_path", type=str, help="path to output, directly to blob"
    )
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs("cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    dir_path = args.dir_path
    output_blob_path = args.output_blob_path
    pre_train_weights_path = os.path.join(
        dir_path, "models", "noisy.student.notop-b0.h5"
    )

    print(
        f"Is pre_train_weights_path valid? : {os.path.exists(pre_train_weights_path)}"
    )
    print(f"Pre_train_weights_path: {pre_train_weights_path}")

    print("dir_path", "=" * 50)
    print(dir_path)
    print("output_blob_path", "=" * 50)
    print(output_blob_path)


def subtract(a, b):
    return (a - b) * 10
