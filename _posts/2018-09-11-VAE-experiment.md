---
title: CVAE experiment using MNIST
key: 2018091101
tags: DeepLearning GenerativeLearning Auto-encoder Variational Bayesian
category: blog
---

## Conditional Variational Autoencoder

- z dimension: 2
- n layers: 3

당연히 condition이 없을때보다 loss값이 더 낮음. Label에 상관없이 training sample이 잘 분포한 것을 볼 수 있음.

![](/assets/images/cvae_exp/cvae.png)

condition이 없을 때보다 reconstruction도 월등함.

![](/assets/images/cvae_exp/cvae_valid.png)

생각보다 latent space의 2-dimension traversal에서 명확한 feature가 보이지 않음.

![](/assets/images/cvae_exp/cvae_travel-1.png)

![](/assets/images/cvae_exp/cvae_travel-2.png)

![](/assets/images/cvae_exp/cvae_travel-5.png)

![](/assets/images/cvae_exp/cvae_travel-9.png)

## Condition Embedded Variational Autoencoder

- z dimension: 2
- n layers: 3
- emb weight: 10 --> 2

One-hot vector에 linear layer를 거친 후에, encoder와 decoder에 들어가도록 수정함. 아래의 그림과 같이 기존의 CVAE와 다르게 비슷한 모양의 sample들끼리 cluster를 이루고 있는 것을 볼 수 있음. 이것으로 기대하는 효과는 one-hot 벡터와 다르게 다른 숫자로부터도 좀 더 배우길 바랬음.

![](/assets/images/cvae_exp/emb_cvae.png)

![](/assets/images/cvae_exp/emb_cvae_valid.png)

embedding vector를 condition으로 받기 때문에, 고정된 숫자를 condition으로 주더라도, traversal할 때에 여러개의 숫자가 나오는 것을 확인할 수 있다.

![](/assets/images/cvae_exp/emb_cvae_travel-1.png)

![](/assets/images/cvae_exp/emb_cvae_travel-2.png)

![](/assets/images/cvae_exp/emb_cvae_travel-3.png)