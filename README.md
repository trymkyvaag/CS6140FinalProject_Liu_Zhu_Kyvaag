# CS6140FinalProject_Liu_Zhu_Kyvaag
3D dataset: https://www.synapse.org/Synapse:syn53708249/files/
2D dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset


Here is instructions on how to run both the 2D and 3D dataset.
Instructions are expected to be run in colab or similar but only thing that would have to be changed is datapathh:

Colab:  

!git clone https://github.com/trymkyvaag/CS6140FinalProject_Liu_Zhu_Kyvaag.git  
!cd CS6140FinalProject_Liu_Zhu_Kyvaag  

from google.colab import drive  
drive.mount('/content/drive')     
!cp -r /content/drive/MyDrive/data /content/data #Assuming dataset is in drive  
!python /content/CS6140FinalProject_Liu_Zhu_Kyvaag/notebooks/models/run.py --epochs 20 --batch_size 64  


Or in terminal:  

python3 /content/CS6140FinalProject_Liu_Zhu_Kyvaag/notebooks/models/run.py --epochs 20 --batch_size 64 #but change the datapath in run.py main to whereever it is downloaded.  




3d: 
run notebook after checking that datapaths are correct for where your data is downloaded.

