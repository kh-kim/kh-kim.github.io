---
title: Back-Translation Review
key: 20200930
tags: NaturalLanguageProcessing
category: blog
---

# Back-Translation Review

이번 포스팅은 Neural Machine Translation (NMT)에서 널리 사랑받고 있는 Back-Translation (이하 BT)에 대해서 좀 더 이해해보는 시간을 가지려 합니다.
기존에 제안된 BT에 대해서 살펴보고, 기본 BT의 한계를 극복하기 위해 제안된 여러가지 방법들을 여러가지 관점(실험 + 수식)에서 이해해보고자 합니다.

## Leverage with Monolingual Corpus

NMT는 2014년 Sequence-to-Sequence의 발명 이후로 자연어생성(NLG) 분야와 함께 큰 발전을 이루어왔습니다.
특히 제가 보았을 때, 2017년 Transformer의 발명 덕분인지, 2018년에 거의 연구의 정점을 찍은 것으로 보입니다.
즉, 구성 요소들이 잘 갖춰진 문장(e.g. 뉴스기사)들에 대해서는 이미 사람의 수준을 넘어섰다고 할 정도의 성능에 이르렀습니다.

하지만 NMT 개발을 위해서는 두 쌍의 언어가 문장 수준에서 mapping되어 있는 parallel corpus가 필수적으로 필요하고, 이는 여전히 low-resource 상황에서의 NMT에서 큰 장벽이 되고 있습니다.
예를 들어, 한국어-영어 번역은 매우 잘 연구/개발되어 있지만, 한국어-태국어와 같은 번역은 이에 반해 훨씬 미비한 상황입니다.
더욱이 인터넷 등에서 무한대로 모을 수 있는 unlabled corpus에 반해, parallel corpus는 매우 수집이 어렵고 비싸기 때문에, 한국어-영어 번역에서도 여전히 parallel corpus 수집에 대한 목마름은 항상 남아있습니다.

따라서 예전부터 단방향 코퍼스(monolingual corpus)를 활용하여 번역기의 성능을 높이고자 하는 시도들은 매우 많았고, 개인적으로도 굉장히 좋아하는 주제라고 생각합니다. -- 기계번역의 꽃이랄까요.
Language Model Ensemble[x et al., 9999]에서부터 Dual Learning[x et al., 9999]에 이르기까지 정말 많은 연구들이 있었고, 모두 parallel corpus만을 활용한 것보다 더 나은 성능을 제공할 수 있었습니다.
하지만 BT는 굉장히 이른 시기에 제안되었음에도 불구하고, 간단한 방법으로 비교적 훌륭한 결과물을 제공하기 때문에 위의 여러가지 방법들 중에서도 가장 사랑받는 방법 중에 하나였습니다.
이 포스팅에서는 Back-Translation에 대해서 살펴보고, 여러가지 관점에서 살펴보고자 합니다.

## Back-Translation

Back-Translation은 에딘버러 대학의 리코 센리치 교수가 제안한 방법으로, 센리치 교수님은 BPE를 통한 subword segmentation을 제안한 분으로도 유명합니다.
Parallel corpus의 부족으로 인해 겪는 가장 기본적인 문제중에 하나는, 디코더인 타깃 언어의 언어모델(Language Model, LM)의 성능 저하를 생각해볼 수 있습니다.
즉, 다량의 monlingual corpus를 수집하여 풍부한 표현을 학습할 수 있는 언어모델에 비해, parallel corpus만을 활용한 경우에는 훨씬 빈약한 표현만을 배울 수 밖에 없습니다.
따라서, 소스 언어 문장으로부터 타깃 언어 문장으로 가는 translation model(TM)의 성능 자체도 문제가 될테지만, 번역에 필요한 정보를 바탕으로 완성된 문장을 만들어내는 능력도 부족할 것 입니다.

이때, TM의 성능 저하는 parallel corpus의 부족과 직접적으로 연관이 있지만, LM의 성능 저하는 monolingual corpus를 통해 개선을 꾀해볼 수 있을 것 같습니다.
하지만 예전 Statistical Machine Translation (SMT)의 경우에는 보통 TM과 LM이 명시적으로 따로 존재하였기 때문에 monolingual corpus를 통한 LM의 성능 개선을 쉽게 시도할 수 있었지만, NMT에선 end-to-end 모델로 이루어져 있으므로 LM이 명시적으로 분리되어 있지 않아 어려움이 있습니다.
BT는 이러한 상황에서 디코더의 언어모델의 성능을 올리기 위한 (+ 추가적으로 TM의 성능 개선도 약간 기대할 수 있는) 방법을 제안합니다.

보통 번역기를 개발할 경우, 한 쌍의 번역 모델이 자연스럽게 나오게 됩니다.
왜냐하면 우리는 parallel corpus를 통해 번역기를 개발하므로, 두 방향의 번역기를 학습할 수 있기 때문입니다.
이때 Back-Translation이라는 이름에서 볼 수 있듯이, BT는 반대쪽 모델을 타깃 모델을 개선하는데 활용합니다.

예를 들어 아래와 같이 parallel corpus $\mathcal{B}$ 와 monolingual corpus $\mathcal{M}$ 을 수집한 상황을 생각해볼 수 있습니다.

$$\begin{gathered}
\mathcal{B}=\{(x_n, y_n)\}_{n=1}^N \\
\mathcal{M}=\{y_s\}_{s=1}^S
\end{gathered}$$

그럼 자연스럽게 우리는 일단은 $\mathcal{B}$ 를 활용하여 두 개의 모델을 얻을 수 있습니다.

$$\begin{aligned}
\hat{\theta}_{x\rightarrow{y}}&=\underset{\theta\in\Theta}{\text{argmax}}\sum_{n=1}^N{\log{P(y_n|x_n;\theta_{x\rightarrow{y}})}} \\
\hat{\theta}_{y\rightarrow{x}}&=\underset{\theta\in\Theta}{\text{argmax}}\sum_{n=1}^N{\log{P(x_n|y_n;\theta_{y\rightarrow{x}})}} \\
\end{aligned}$$

즉, 

### 성능

### 한계

## Noise 추가를 통한 Back-Translation 개선

## Tagged Back-Translation

## 실제 Back-Translation은 효과가 있을까?

## Back-Translation 수식으로 풀어보기

### 기존 방법들을 수식 관점에서 이해하기

## 정리하며

## 참고문헌

