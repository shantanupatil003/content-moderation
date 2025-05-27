# Create a new conda environment with Python 3.10
conda create -n content-moder python=3.10
2
# Activate the environment
conda activate content-moder

# Install the required packages
pip install pydub
pip install ffmpeg-python
pip install requests
pip install numpy
pip install pandas


# USE requirements 
pip install -r requirements.txt