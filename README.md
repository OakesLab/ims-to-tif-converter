# ims-to-tif-converter
Converts Fusion IMS files to TIF files

Requirements:
sys
PIL (pillow)
scikit-image
numpy
h5py
os
glob

Usage:
1. Navigate to a directory containing only IMS files. The script will abort if
   there are any .tif files present.
2. Run the script with "python <full_path_to_script> <downsample_level>" where
   <full_path_to_script> should be replaced with the full path to where you
   saved the driver script, and <downsample_level> should be replaced with the
   downsample ratio that you desire. The downsample level must be a factor
   of two, and divide evenly into the dimensions of your image. 

Use 1 for your downsample ratio if you want a straight conversion. Otherwise,
the dimensionality of your image will be divided by the supplied downsample
level in the output.
