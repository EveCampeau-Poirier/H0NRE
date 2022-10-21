import argparse
from functions import inference, joint_inference


if __name__ == "__main__":

    # --- Training ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Compute posteriors and coverage diagnostic")
    parser.add_argument("--keys_file", type=str, default="", help="path and name of the key file")
    parser.add_argument("--data_file", type=str, default="dataset.hdf5", help="path and name of the dataset")
    parser.add_argument("--model_file", type=str, default="", help="path and name of the model")
    parser.add_argument("--path_out", type=str, default="", help="path to save the outputs")
    parser.add_argument("--joint", type=bool, default=False, help="joint inference")
    args = parser.parse_args()
    
    if args.joint:
        joint_inference(args.data_file, args.model_file, args.path_out)
    
    else:
        inference(args.keys_file, args.data_file, args.model_file, args.path_out)