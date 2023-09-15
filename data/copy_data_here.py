import os 
import subprocess
import glob 
import copy 

import shutil




path = '/home/kinova/john/data/bop_2023_diff_dope/ycbv/test/'
path_out = 'ycbv/test/'

path = '/home/kinova/john/data/bop_2023_diff_dope/tless/test_primesense/'
path_out = 'tless/test_primesense/'

path = '/home/kinova/john/data/bop_2023_hope_paper/hope/val/'
path_out = 'hope/val/'


wildcard='*error*010*'



all_scene_names = list(sorted(os.listdir(path)))

for scene_name in all_scene_names:
    scene_path = f"{path}/{scene_name}"

    # Figure out what json files we have that match the wildcard
    scene_json_wcard = f"{scene_path}/{wildcard}"
    all_json_paths = glob.glob(scene_json_wcard)
    for json_file in all_json_paths:
    	path_name = os.path.dirname(json_file).replace(path,path_out)
    	os.makedirs(path_name,exist_ok=True)

    	print(json_file)
    	shutil.copyfile(json_file, f"{path_name}/{os.path.basename(json_file)}")
