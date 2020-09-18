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

## Back-Translation이란

### 성능

### 한계

## Noise 추가를 통한 Back-Translation 개선

## Tagged Back-Translation

## 실제 Back-Translation은 효과가 있을까?

## Back-Translation 수식으로 풀어보기

### 기존 방법들을 수식 관점에서 이해하기

## 정리하며

## 참고문헌

