input_type = 3  #   3 with input: sand,silt,clay,bulk density
# hydrualic_Model = 1  #  1 = VGM;    2 = FXW-M3;  3 = B-FXW

import os
import subprocess
import sys

base_dir = os.path.dirname(os.path.abspath(__file__)) # path
model_dir = os.path.join(base_dir, "FNN_Model")
csv_path = os.path.join(base_dir, "texture.csv")

Model_name = ['VGM','FXW_M3','B_FXW']

for hydrualic_Model in range(3):
    script_name = Model_name[hydrualic_Model] + '_PTF.py'  # make 'modelname+_PTF.py'
    script_path = os.path.join(model_dir, script_name)

    output_name = Model_name[hydrualic_Model] + '_Parameter.csv'  # make 'modelname+.csv'
    output_path = os.path.join(base_dir, output_name)

    subprocess.run([
        sys.executable,
        script_path,
        "--input", csv_path,
        "--output", output_path
    ], check=True)
    print("Finish")
