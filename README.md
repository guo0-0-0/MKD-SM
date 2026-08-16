# MKD-SM
This is the official implementation of the paper "Multi-Slice Knowledge-Driven System Matrix Calibration in Magnetic Particle Imaging".

## Important Dependencies:
- Python == 3.9.13
- PyTorch == 2.0.1
- NumPy == 1.19.5

## Data Preprocessing
You should first download the raw .mdf data from the openMPI website: 
https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/index.html

Please download the Calibration Measurements 7 and 10, and put them in the OpenMPI_SM/ folder.

Make sure the following file structure:

--OpenMPI_SM
----OpenData_7.mdf
----OpenData_10.mdf

Then you can run the following command to preprocess the data:

- python data_process.py

## Train
After data preprocessing, you can run the following command to train the model:

- python train_MKD_SM.py

## Predict
After training, you can run the following command to predict the system matrix:

- python evaluate_MKD.py

## Reference
If you take advantage of this paper in your research, please cite the following in your manuscript:

- @ARTICLE{11267072,
  author={Guo, Pengyue and Wei, Zechen and Zeng, Yu and Wang, Bingye and Liao, Yidong and Hu, Jiawei and Hou, Lingwen and Liu, Kai and He, Ning and Wang, Qibin and Li, Lei and Hui, Hui and Wang, Yihan and Zhu, Shouping and Tian, Jie},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Multi-Slice Knowledge-Driven System Matrix Calibration in Magnetic Particle Imaging}, 
  year={2026},
  volume={12},
  number={},
  pages={25-36},
  doi={10.1109/TCI.2025.3636749}}
