## Paper Title ##
**GARNN: An Interpretable Graph Attentive Recurrent Neural Network for Predicting Blood Glucose Levels via Multivariate Time Series**
Chengzhe Piao, Taiyu Zhu, Stephanie E Baldeweg, Paul Taylor, Pantelis Georgiou, Jiahao Sun, Jun Wang, Kezhi Li*
## Datasets ##
We introduce four datasets to our experiments, i.e., OhioT1DM, ArisesT1DM, ShanghaiT1DM, ShanghaiT2DM.

All datasets can be accessed publicly apart from ArisesT1DM, which can be accessed via authorised procedures by contacting the project manager and the corresponding author. ArisesT1DM was collected following applicable legal standards, thereby eliminating the need for additional ethical clearance for this study.

* **OhioT1DM**: Marling, C.; and Bunescu, R. C. 2020. The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020. In KDH@ECAI’20, volume 2675, 71–74
* **ArisesT1DM**: Zhu, T., Uduku, C., Li, K., Herrero, P., Oliver, N., & Georgiou, P. (2022). Enhancing self-management in type 1 diabetes with wearables and deep learning. npj Digital Medicine, 5(1), 78.
* **ShanghaiT1DM; ShanghaiT2DM**: Zhao, Q., Zhu, J., Shen, X., Lin, C., Zhang, Y., Liang, Y., ... & Wang, C. (2023). Chinese diabetes datasets for data-driven machine learning. Scientific Data, 10(1), 35.

### Data Preprocessing

**NOTE: Please revise the paths in these codes before running them!**

After getting these datasets, please run the codes under the folder "/gen_datasets" to preprocess the data. 

### Training and Testing

**NOTE: Please revise the paths in these codes before running them!**

Please run "run_XXXX.py", then all experiments can be repeated.

We used the following packages with RTX 3090 Ti to run the codes:
* PyTorch 1.11.0
* Scikit-Learn 1.0.2
* Numpy 1.21.5
* Scipy 1.7.3
* Pandas 1.4.2 

End


