import json
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("lang", type=str, default="de")
args = parser.parse_args()


# first move the folder of the annotator whose overlapping files were chosen to be included into annotatinos_final/lang/ 
# then change path in this script to the annotations whose overlapping files were not chosen to be included 
# then run this script to complete the final annotations 

def main():
    for file in os.listdir(f"annotations_raw/annotations2/{args.lang}/"):
        if file in os.listdir(f"annotations_final/{args.lang}/"):
            continue
        else:
            # Copy file to final annotations directory
            src = f"annotations_raw/annotations2/{args.lang}/{file}"
            dst = f"annotations_final/{args.lang}/{file}"
            shutil.copy(src, dst)


if __name__ == "__main__":
    main()