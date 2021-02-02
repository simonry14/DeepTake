# [DeepTake:Prediction of Driver Takeover Behavior using Multimodal Data](https://arxiv.org/abs/2012.15441)
[Erfan Pakdamanian](https://www.cs.virginia.edu/~ep2ca/index.html), [Shili Sheng](https://www.researchgate.net/profile/Shili_Sheng) ,[Sonia Baee](http://soniabaee.com),[Seongkook Heo](https://seongkookheo.com), [Sarit Kraus](https://u.cs.biu.ac.il/~sarit/), [Lu Feng](https://www.cs.virginia.edu/~lufeng/)

DeepTake framework Preview:

[![DeepTake preview](https://img.youtube.com/vi/clw8O1t1Zms/0.jpg)](https://www.youtube.com/watch?v=clw8O1t1Zms)


## Overview
![DeepTake_Overview](https://github.com/erfpak7/DeepTake/tree/main/preview/DeepTake-overview-cropped.png)

Automated vehicles promise a future where drivers can engage in non-driving tasks without hands on the steering wheels for a prolonged period. Nevertheless, automated vehicles may still need to occasionally hand the control back to drivers due to technology limitations and legal requirements. While some systems determine the need for driver takeover using driver context and road condition to initiate a takeover request, studies show that the driver may not react to it. We present DeepTake, a novel deep neural network-based framework that predicts multiple aspects of takeover behavior to ensure that the driver is able to safely take over the control when engaged in non-driving tasks. Using features from vehicle data, driver biometrics, and subjective measurements, DeepTake predicts the driver's intention, time, and quality of takeover. We evaluate DeepTake performance using multiple evaluation metrics. Results show that DeepTake reliably predicts the takeover intention, time, and quality, with an accuracy of 96%, 93%, and 83%, respectively. Results also indicate that DeepTake outperforms previous state-of-the-art methods on predicting driver takeover time and quality. Our findings have implications for the algorithm development of driver monitoring and state detection.


## DeepTake Framework
![DeepTake_Framework](https://github.com/erfpak7/DeepTake/tree/main/preview/NN_Structure.png)
The neural network is a fully-connected feed-forward classifier with three hidden layers as shown in the figure. DeepTake utilizes a feed-forward DNN with a mini-batch stochastic gradient descent. The DNN model architecture begins with an input layer to match the the number of input features, and each layer receives the input values from the prior layer and outputs to the next one. DeepTake uses three hidden layers with 23, 14, and 8 ReLu units. 



## Citing
If you find this paper/code useful, please consider citing our paper:
```bash
@article{pakdamanian2020deeptake,
  title={DeepTake: Prediction of Driver Takeover Behavior using Multimodal Data},
  author={Pakdamanian, Erfan and Sheng, Shili and Baee, Sonia and Heo, Seongkook and Kraus, Sarit and Feng, Lu},
  journal={arXiv preprint arXiv:2012.15441},
  year={2020}
}
```


