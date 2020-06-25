---
title: Introduction to Deep Time-series Anomaly Detection
key: 20200624
tags: AnomalyDetection
category: blog
---

# Introduction to Deep Time-series Anomaly Detection

이번 포스팅에서는 시계열(time-series) 또는 시퀀셜(sequntial) 데이터에 대한 anomaly detection 기법을 이야기하고자 합니다. 사실 시계열 데이터는 데이터의 발생 시점(e.g. interval)도 중요한 feature인데 반해, 단순한 시퀀셜 데이터는 보통 순서 정보만 활용됩니다. 하지만 딥러닝의 대부분의 모델들은 기본적으로 시퀀셜 데이터만 다루도록 설계되어 있습니다. 이 포스팅은 딥러닝을 활용한 일반적인 방법론에 대해 이야기하고자 하므로, 시계열 데이터의 샘플간 time interval이 동일하다고 가정하고 seasonality와 같은 issue는 배제하고 이야기하고자 합니다. 즉, 비록 시계열 데이터이긴 하지만 일반적인 시퀀셜 데이터와 같이 다루도록 하겠습니다. 추후 다른 포스팅에서 interval 또는 seasonality와 같은 issue에 대해서 다루도록 하겠습니다.

## Previous Methods

사실 이상탐지의 대부분 application들은 의외로 시계열 데이터인 경우가 많습니다. Computer vision 분야에서의 image anomaly detection과 같은 case를 제외한다면, 특히 numeric data 또는 tabular data인 경우 대부분이 시계열 데이터로 구성되어 있는 경우가 많습니다. 따라서 예전부터 이상탐지 분야에서 시계열 데이터에 대한 이상탐지에 대한 연구가 많이 이루어져 왔습니다. 하지만 이에 반해, 현재 딥러닝에서의 이상탐지 연구들은 대부분 iid 기반이 많은 것이 사실입니다.

### Univariate Time-series Anomaly Detection

![심장박동](https://upload.wikimedia.org/wikipedia/commons/b/bd/12leadECG.jpg)

보통 시계열 이상탐지 연구는 univariate time-series 데이터에 대해서 연구되어왔습니다. Image에서의 이상탐지라고 한다면, 주어진 이미지가 정상 범위에서 벗어날 경우 이것에 대해서 탐지하는 문제가 될 것입니다. 이에 반해 시계열 이상탐지 문제는 주어진 기간동안의 신호들이 정상 범위에서 벗어날 경우 이것에 대해서 탐지할 수 있어야 합니다. 예를 들어 심장 박동의 이상을 탐지하는 문제라고 한다면, 주어진 시간동안의 심장의 움직임에 대한 신호들을 가지고 해당 시간 내에 비정상적인 심장의 움직임이 있었는지 탐지하는 문제가 될 것입니다.

딥러닝 이전에는 DTW(Dynamic Time Warping)이나 ARIMA를 활용한 방법들도 많이 연구되었습니다. 딥러닝 모델을 활용해서도 여러가지 연구가 진행되었으나, 오늘의 주요 주제는 아니므로 넘어가도록 하겠습니다.

### Multivariate Time-series Anomaly Detection

![MFCC 출처: https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd](https://miro.medium.com/max/1400/1*pzE4i1TXaLCmzTXgdxFZjQ.jpeg)

사실 오늘 주로 다루고자 하는 주제는 multivariate time-series 데이터에 대한 이상탐지입니다. 예를 들어, 위의 그림은 univariate time-series 데이터인 오디오 신호를 [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)로 나타낸 것입니다. 위의 그림에서 x축은 시간을 나타내고, y축은 주파수 대역을 의미합니다. 따라서 특정 시간에 발생한 전체 주파수 대역에서의 신호의 세기가 multivariate vector로 나타내어질 수 있을 것입니다. 그럼 정해진 시간동안의 multivariate vector들의 시퀀스가 주어졌을 때, 해당 벡터들의 시퀀스가 정상 범위 내에 있는지 판단하는 문제가 될 것입니다.

사실 univariate time-series anomaly detection은 심장박동 이상탐지와 같은 문제에서는 훌륭하게 동작할 수 있지만, 많은 문제에 그대로 적용되기에는 어려움이 있습니다. 위의 오디오 신호에 대한 예제를 포함하여, 많은 반례를 생각해볼 수 있습니다. 예를 들어 제가 [DEVIEW 2019에서 발표](https://deview.kr/2019/schedule/286)할 때 데모로 소개해드렸던 로봇팔 이상탐지의 경우도 해당 될 수 있습니다.

해당 로봇팔은 6개의 축을 가지고 있습니다. 즉, 6개의 전기모터로 구성되어 있고, 이 모터에 들어가는 전류값(current)을 신호로 삼아서 이상탐지 문제를 접근해 볼 수 있을 것입니다. 이때, 각 축 별로 univariate time-series 이상탐지 모델을 만들어 적용해볼 수도 있을 것입니다. 하지만 이 경우에는 각 축간의 상호작용은 전혀 파악할 수 없을 것입니다. 예를 들어 1번 축이 높은 값일 때는 다른 축들이 낮은 값을 지녀야 한다던지와 같은 상황에 대해서는 대처할 수 없을 것입니다. 따라서 이러한 문제에서는 multivariate time-series 모델을 도입하여, 각 feature 사이의 상관관계까지도 학습할 수 있습니다.

이때, 기존의 shallow 기법들은 univariate 위에서의 time-series를 다루기에도 벅찬 상황이기 때문에, 딥러닝을 활용한 이상탐지 모델링 기법이 큰 힘을 발휘할 수 있습니다.

## Deep Time-series Anomaly Detection

재미있게도 일찍이 딥러닝 이전의 시절에도 LSTM의 존재는 있었지만, 당시에는 데이터와 컴퓨팅 파워의 부족으로 인해서 부담스러운 존재였던 것도 사실입니다. 하지만 이제는 이전의 문제들이 대부분 해결되어 LSTM 따위는 아무런 부담없이 학습할 수 있게 되었습니다. 따라서 우리가 풀고자 하는 데이터가 시퀀셜 또는 시계열의 성격(샘플이 매번 독립적으로 같은 분포에서 샘플링 되는 것이 아니라면)을 갖고 있다면, RNN 계열의 모델을 활용해 보는 것도 매우 좋은 방법일 것 입니다.

### Using Autoencoders with Flatten Features

### Single RNN based Methods

### Encoder-Decoder based Methods

#### Attention?

#### Teacher Forcing

#### Posterior Collapse

#### Likelihood: Reconstruction Error

#### Generations

##### Other Fields: Natural Language Generations

###### Minimum Risk Training (MRT) in Machine Translations

##### In Anomaly Detection Case

## Conclusion

## References

- [1] [Inference Suboptimality in Variational Autoencoders](https://arxiv.org/pdf/1801.03558.pdf)
- [2] [Semi-Amortized Variational Autoencoders](https://arxiv.org/pdf/1802.02550.pdf)
- [3] [VARIATIONAL AUTOENCODERS FOR TEXT MODELING WITHOUT WEAKENING THE DECODER](https://openreview.net/pdf?id=H1eZ6sRcFm)
- [4] [Re-balancing Variational Autoencoder Loss for Molecule Sequence Generation](https://arxiv.org/pdf/1910.00698.pdf)
- [5] [InfoVAE: Balancing Learning and Inference in Variational Autoencoders](https://arxiv.org/pdf/1706.02262.pdf)
- [6] [LAGGING INFERENCE NETWORKS AND POSTERIOR COLLAPSE IN VARIATIONAL AUTOENCODERS](https://openreview.net/pdf?id=rylDfnCqF7)
- [7] [Variational Attention for Sequence-to-Sequence Models](https://arxiv.org/pdf/1712.08207.pdf)
