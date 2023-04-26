<p align="center">
    <img src="https://github.com/sariahmghames/NeuroSyM-prediction/blob/main/NeuroSyM-sgan/neurosgan_images/logo.jpg" width="490" height="144" /> 
</p> 

This repository applies a neuro-symbolic approach for enhancing multivariate time series prediction accuracy with application to context-aware human motion prediction 


The scientific content of this work can be found @ https://arxiv.org/abs/2304.11740. To cite this work, please refer to the proceedings of the International Joint Conference on Neural Networks (IJCNN 2023), with the following citation:

```
@inproceedings{mghames2023,
  title={A Neuro-Symbolic Approach for Enhanced Human Motion Prediction},
  author={Mghames, Sariah and Castri, Luca and Hanheide, Marc and Bellotto, Nicola},
  booktitle={2023 IEEE International Joint Conference on Neural Networks (IJCNN)},
  pages={},
  year={2023},
  organization={IEEE}
}
```


## Results
The following results applies to zara01 test dataset on sgan framework and its neurosym version. We illustrate one single batch with only 6 pedestrians motion for clarity, though the context is much larger.

<p align="center">
    <img src="https://github.com/sariahmghames/NeuroSyM-prediction/blob/main/NeuroSyM-sgan/neurosgan_images/zara01_gt_8ts_neurosym.gif" width="320" height="240" /> 
</p>

<p align="center">
  <img src="https://github.com/sariahmghames/NeuroSyM-prediction/blob/main/NeuroSyM-sgan/neurosgan_images/zara01_pred_8ts_sgan.gif" width="320" height="240" />
  <img src="https://github.com/sariahmghames/NeuroSyM-prediction/blob/main/NeuroSyM-sgan/neurosgan_images/zara01_pred_8ts_neurosym.gif" width="320" height="240" />
</p>

## Training and Testing
Please refer to the readme.md inside each architecture package

*** We welcome any issue or collaborations. You can reach out @ sariahmghames@gmail.com ***
