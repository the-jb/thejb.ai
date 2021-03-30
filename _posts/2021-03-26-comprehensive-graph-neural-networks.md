---
layout: post
title: Graph Neural Networks 개념 총정리 1 - 개요
tags: [graph, basic knowledge]
---

# 소개

기존에 Graph 개념을 충분히 알고 있고, 알고리즘을 통해 많이 활용해봤지만 Graph Neural Networks 에 대한 개념을 잡기가 쉽지 않았다. 처음 접근할 때 가장 어려웠던 점은 "대체 그래프에서 왜 딥 러닝이 필요한가?"에 대한 감을 잡는 것이었다. 이미지, 언어 등의 분야는 인식, 분석하고자 하는 대상이 명확하다. 하지만 그래프는 그냥 연결 관계 그 자체이다. 그래프를 왜 굳이 딥 러닝으로 인식하려고 하는 것인지, 이를 통해서 어떤 것을 얻을 수 있는지 직관적으로 떠올리기 어려웠다. 결국에 이미지나 언어 등과 달리, 그래프는 목적과 수단이 매우 다양하고 폭넓은 데이터에 적용하는 수단이었다.

이 포스트에서는 GNN에 대한 전반적인 이론들에 대해 어느정도 깊이 있게 이해하기 위해 위 "A Comprehensive Survey on Graph Neural Networks" 라는 survey 논문을 타겟으로 삼았다. 하지만 논문을 그대로 해석하는 것이 아니라 GNN에 대한 이해 측면에 중점을 두어 나름대로 주관적인 해석과 설명을 붙이고, 내용들을 참고하여 여러가지 그래프 문제들과 그 것을 해결하기 위한 다양한 방법들을 어떤 식으로 구분하고 분류했는지 살펴보고자 한다.

# 참고 논문

- [Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, Philip S. Yu, "A Comprehensive Survey on Graph Neural Networks"](https://arxiv.org/abs/1901.00596)

# 도입

최근 딥 러닝의 발전은 이미지, 비디오, 언어 분야에서 많이 이루어졌다. 이 데이터들은 모두 유클리드 공간에서 표현되는 데이터들이다. 반면에 비유클리드 공간의 데이터들은 오브젝트 간의 복잡한 관계나 상호의존성등을 정의할 수 있는 그래프로 표현할 수 있다.

이 논문에서는 Graph Neural Networks(GNN)을 다음과 같이 분류를 한다. 

1. Namely recurrent graph neural networks
2. Convolutional graph networks
3. Graph autoencoders
4. Spatial-temporal graph neural networks

# 유클리드 공간과 비유클리드 공간

딥 러닝에서 가장 많이 활용되는 기법들로 CNN, RNN, autoencoder 등이 있다. 특히 이러한 기법들이 빠르게 발전한 이유중 하나는 GPU 등으로 많은 데이터에 대해서 빠른 계산이 가능해졌기 때문이다. 그리고 이러한 계산들은 특히 유클리드 데이터(이미지, 텍스트, 비디오)의 latent representations 를 추출하기 유리하다. 예를 들어서 이미지 데이터는 유클리드 공간의 격자(grid)로 표현이 가능하다. 이미지에 CNN을 활용하면 shift-invariance, local connectivity, compositionality 가 가능하다. 따라서 CNN이 이미지에서 부분적으로 중요한 feature들을 뽑아낼 수 있고, 이를 통해 전체적인 이미지 분석을 효과적으로 할 수 있다.

반면에 비유클리드 공간은 보통 그래프로 많이 표현된다. 쇼핑에서의 상품 추천, 화학에서 분자 구조, 사람들간의 관계 네트워크 등이 이에 포함된다. 그래프는 노드의 사이즈도 다양하고, 순서관계도 없고, 이웃의 크기도 다른 등 데이터가 규칙적이지 않다. 그래서 이를 표현하거나, 아니면 그래프로 어떤 계산을 하는 자체가 쉽지 않다. 특히 현재 머신 러닝에서는 기본적으로 인스턴스들이 서로 독립적이어야 한다는 전제가 있다. 하지만 그래프에서 노드는 서로 독립적이지 않고, 다른 노드들과 여러 관계를 맺고 있어서 기존 v유클리드 공간에서 활용하던 머신러닝 기법들을 적용하기 어렵다.

이렇게 비유클리드 공간, 즉 그래프 분석을 위해서 최초의 GNN 논문 이후로 많은 연구가 나오게 된다.

# Graph Neural Networks와 유사한 개념 정리

## Network Embedding 의 차이

GNN 연구에 있어서 Network embedding은 밀접한 관련이 있다. 하지만 둘 사이의 개념은 차이가 있다.

Network embedding 은 네트워크 노드를 낮은 차원의 벡터로 표현하는 것이다. Network embedding 은 노드를 표현하는 차원을 줄이면서도 노드 구조와 노드들의 내용을 보존할 수 있도록 하는데 중점을 둔다. 반면에 Graph Neural Networks 의 목적은 그래프 표현 자체가 아니고 그래프를 이용해서 어떤 업무를 수행하거나, 문제를 푸는 것에 있다.

즉, 어떤 문제를 풀기 위해서 GNN은 문제를 푸는 모델 자체가 되는 반면, Network embedding 은 이 임베딩을 통해 표현된 값들을 바탕으로 문제를 푸는 여러가지 모델을 적용하는 개념이라는 점에서 가장 큰 차이가 있다.

## Graph Kernel Methods 의 차이

Graph kernel methods 는 그래프 분류 문제를 풀기 위한 주요 기법이었다. 이 기법들은 kernel 함수를 통해 그래프간의 유사도를 측정하고, 이를 이용해서 kernel-based 알고리즘들을 사용해서 그래프를 학습한다.

GNN 과 비슷하게 graph kernel은 그래프나 노드를 mapping function 을 통해 임베딩해서 사용할 수 있다. 여기서 GNN과의 차이점은 이 mapping function 이 결정되어 있냐, 아니면 학습가능하냐이다. Graph kernel methods 는 유사도 계산을 따로 하기 때문에 계산에 병목이 생기게 된다. 하지만 GNN 은 전체 모델 자체가 임베딩부터 분류까지 한번에 이루어지기 때문에 학습에 훨씬 효과적이다.

# 그래프에 대한 정의

이후 내용들을 위해 그래프에 대한 식과 변수들을 여기서 먼저 정의한다.

- 그래프
  - 그래프란 $G=(V,E)$ 로 표현할 수 있다.
  - $V$ 는 정점(노드)의 집합을 나타내고, $E$는 간선(edge)의 집합을 나타낸다.
  - $v_i, v_j\in V$일 때, 엣지를 $e_{ij}=(v_i,v_j)\in E$ 로 나타낼 수 있다. 여기서 $e_{ij}$는 $v_j$에서 $v_i$로 연결되는 엣지를 의미한다. 
  - $v$의 이웃 노드를 $N(v)=\{u\in V\vert (v, u) \in E\}$ 로 나타낸다.
  - 그래프의 인접 행렬을 $A:n\times n$으로 표현한다. $e_{ij}\in E$일 때, $A_{ij}=1$이고 아닐 경우 $A_{ij}=0$이다.
  - node attributes를 $X \in R^{n\times d}$라고 하며, $x_v\in R^d$는 노드$v$의 feature vector 를 나타낸다.
  - edge attributes를 $X^e\in R^{m\times c}$라고 하며, $x^e_{v,u}\in R^c$는 $(v,u)$엣지에 대한 feature vector를 나타낸다.
- 방향성 그래프
  - 방향성 그래프는 그래프에서 모든 엣지가 방향을 띄고 있는 그래프를 말한다.
  - 여기서 무방향 그래프는 방향성 그래프의 특별 케이스(엣지가 양방향으로 동일)로 생각하도록 한다.
  - 그래프가 방향성이 없다면, 인접 행렬은 대칭이 된다.
- Spatial-Temporal(공간적-시간적) 그래프
  - spatial-temporal 그래프는 node attribute 가 시간에 따라 동적으로 변경되는 그래프를 말한다.
  - spatial-temporal 그래프는 $G^{(t)}=(V,E,X^{(t)})$로 정의한다.

# Graph Neural Networks의 속성 익히기

## Graph Neural Networks 모델 구조의 분류

이 논문에서는 Graph Neural Networks 를 다음과 같이 4개의 카테고리로 분류하는 방법을 제시했다. 각 카테고리에 대한 간단한 개념과 목적은 다음과 같다. 이후 포스트에서 각 모델에 대한 자세한 내용을 다뤄볼 것이다.

### 1. Recurrent Graph Neural Networks(RecGNNs)

Graph Neural Networks 가 recurrent 모델로부터 출발하였다. 그래프 자체가 시작과 끝이 있는 것이 아닌, 순환 구조이기 때문에 이 구조를 recurrent 모델을 통해 표현하고자 한 것으로 생각된다.

RecGNN은 노드 표현을 RNN으로 학습하고자 하는 것에 중점을 둔다. 여기서는 노드가 stable equilibrium에 도달할 때 까지 계속해서 정보/메시지를 이웃 노드와 교환한다고 전제하였다.

RecGNN은 다음에 등장하는 ConvGNN에 많은 영향을 주었다는 점에서 의미가 있다. 특히 이 메시지 패싱이라는 아이디어는 spatial-based convolutional GNN 에서도 이어졌다.

### 2. Convolutional Graph Neural Networks(ConvGNNs)

ConvGNN 은 convolution 이라는 연산을 일반 grid data 가 아닌, 그래프 데이터에 적용하는 방법을 정립하였다. ConvGNN 의 핵심 아이디어는 바로 노드 $v$를 그 노드의 feature인 $x_v$와 이웃 노드의 feature인 $x_u,u\in N(v)$를 aggregate 하여 표현했다는 점이다. RecGNN과 다른 점은 ConvGNN은 여러개의 graph convolutional layer 를 쌓아서 노드를 고차원으로 표현했다는 점이다.

ConvGNN 은 여러 종류의 GNN 모델을 설계할 때 가장 중심적으로 사용된다. 예를 들어, graph convolution 사이에 ReLU 를 두어 node classification 모델을 만들 수도 있고, convolution 을 계속 pooling 하여 softmax 를 적용하면 graph classification 모델이 된다.

### 3. Graph Autoencoders(GAEs)

GAE 는 노드나 그래프를 인코딩하고 다시 복원하는 과정에서 latent vector 를 얻는 비지도 학습이다. GAE는 네트워크 임베딩과 graph generative distributions 를 학습하기 위한 용도로 사용된다. 네트워크 임베딩에서 GAE는 latent node representation 을 학습한다. 

### 4. Spatial-temporal Graph Neural Networks(STGNNs)

Spatial-temporal(공간적-시간적) 그래프의 숨겨진 패턴들을 학습하기 위한 네트워크다. 교통량 예측, 사람 동작 인식, 운전 행동 예측 등 다양한 분야에서 중요한 요소가 되고 있다. STGNNs의 핵심 아이디어는 공간적인 의존성과 시간적인 의존성을 동시에 고려하는 것이다. 현재 많은 접근방식들은 graph convolution을 통해 공간적 의존성을, RNN이나 CNN을 통해 시간적 의존성을 가져가는 방식을 사용하고 있다.

## Graph Neural Networks의 output

GNN을 어떻게 활용하는지 살펴보기 위해 먼저 GNN의 output에 대해 봐야 한다. GNN의 출력 형태는 고정되어있지 않고, 다음과 같이 어떤 부분에 초점을 맞추는지에 따라 달라진다.

- **Node-level outputs**

  노드에 대한 regression 혹은 classification 등을 처리하고자 할 때, Node-level outputs를 사용한다. RecGNNs와 ConvGNNs가 고수준의 node representation을 information propagation/graph convolution을 통해 뽑을 수 있다. 

- **Edge-level outputs**

  엣지에 대한 classification, 연결 예측 등에 활용되는 output이다. 두 노드의 hidden representation을 GNN의 input으로 하고, 유사도 함수 혹은 NN을 통해서 엣지의 연결 강도나 label 등을 예측할 수 있다.

- **Graph-level outputs**

  그래프 수준의 output은 어떤 그래프를 분류하는 데 보통 활용된다. 그래프 수준에서의 representation을 얻기 위해 GNN에 pooling이나 readout 등을 결합해서 활용하기도 한다. 이후에 이에 대해 자세히 다루도록 한다.

## Graph Neural Networks의 학습 방법

GNN을 지도학습, 비지도학습 등 다양한 방법으로 학습시킬 수 있다. 이는 학습 목적과 label 정보 유무 등에 따라 다음과 같은 방법 등이 있다.

- **Semi-supervised learning for node-level classification**

  네트워크의 일부 노드들은 label이 되어있고, 다른 노드들이 label되어있지 않은 상태라고 하면, ConvGNN을 통해 unlabeled 노드에 대해 분류하는 모델을 학습시킬 수 있다.

  Graph convolutional layer를 쌓고 나서 multi-class classification을 위한 softmax layer를 연결하면 된다.

- **Supervised learning for graph-level classification**

  그래프 수준에서의 classification은 전체 그래프에 대한 클래스 label을 예측하기 위해 사용한다.

  이를 위한 end-to-end learning은 graph convolutional layer, graph pooling layer, (readout layer)를 사용하면 된다. 여기서 graph convolutional layer는 고수준의 node representation에 필요하고, graph pooling layer는 그래프를 downsampling 하여 각 그래프를 sub-structure로 만드는데 사용한다. readout layer는 node representation들을 graph representation으로 만든다.

- **Unsupervised learning for graph embedding**

  그래프에서 class label 이 따로 없는 경우에는 graph embedding을 비지도학습을 통해 학습시킬 수 있다.

  이 알고리즘들에서 엣지수준의 정보를 두 가지 방법으로 활용한다. 첫 번째로는 autoencoder를 활용하는 방법이다. 여기서 encoder는 graph convolutional layer를 활용하여 구성되고, 그래프를 embedding 시킨다. decoder는 다시 그래프 구조를 재구성한다.

  두 번째로, negative sampling 방법으로, 노드의 쌍의 일부는 negative로, 나머지는 positive로 샘플링하는 것이다. 그리고 logistic regression layer를 통해 postive 쌍과 negative 쌍을 구분한다.

여기까지 GNN의 전체적인 영역과 기본적인 개념들에 대해 살펴보았다. 다음 글 부터는 각 내용에 대해 하나씩 자세하게 다루도록 한다.