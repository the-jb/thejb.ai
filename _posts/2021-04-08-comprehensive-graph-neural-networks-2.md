---
layout: post
title: Graph Neural Networks 개념 총정리 2 - Recurrent GNN
tags: [graph, basic knowledge]
---

# 소개

Recurrent Graph Neural Network 는 GNN의 시초로서 의미가 있다. 과거에는 컴퓨터의 연산 능력의 한계로 주로 방향성 그래프에 대해서만 연구되었다. RecGNN은 고주순의 node representation을 추출하기 위해 노드들의 파라매터들을 재귀적으로 적용한다.

# 최초의 Graph Neural Network 모델

최초의 Graph Neural Network 논문[^1]에서 제안한 가장 기본적인 모델도 RecGNN 에 포함된다. 이 GNN 논문에서는 노드의 hidden state를 다음과 같이 재귀적으로 업데이트한다.

$$
h_v^{(t)}=\sum_{u\in N(v)}f(x_v,x^e(v,u),x_u,h_u^{(t-1)})
$$

여기서 $f$는 결국 어떤 매개변수를 가진 함수, 즉 쉽게 얘기해서 뉴럴넷을 나타내고, $h_v^{(0)}$은 랜덤하게 초기화된다. 여기서 합계 연산을 통해 모든 노드에 GNN을 적용할 수 있다.

이 공식이 수렴하기 위해서는 재귀 함수인 $f$가 축약 매핑(contraction mapping)이 되어야  한다는 조건이 있다. 어쨌든 수렴이 된다고 하면, 마지막 step에서 노드의 hidden state 값들이 readout layer로 들어가게 된다.

[^1]: F. Scarselli, M. Gori, A.Tsoi, M.Hagenbuchner, G.Monfardini, "The Graph Neural Network Model" (2009)

# Gated Graph Neural Network

RNN 에서 자주 사용되는 GRU(Gated Recurrent Unit)을 적용한 모델이다. GGNN에서는 재귀를 고정된 횟수의 step으로 줄이고 있다. 이럴 경우에 더이상 수렴을 위해 파라매터들을 제한시키지 않아도 된다는 장점이 있다. hidden state 공식은 아래와 같다.

$$
h_v^{(t)}=GRU(h_v^{(t-1)},\sum_uWh_u^{(t-1)}), h_v^{(0)}=x_v
$$

GNN과 다른 점은, GGNN에서 BPTT(Back Propagation Through Time)을 사용하여 모델을 학습시킨다는 점이다. 이는 큰 그래프에서는 문제가 될 수도 있는데, GGNN이 모든 노드에 대해 재귀 함수를 여러번 실행하려면 모든 노드의 중간 상태들이 전부 메모리에 저장되어야 하기 때문이다.

# 결론

이 외에도 논문에서는 Graph Echo State Network(GESN), Stochastic Steady-state Embedding(SSE) 등에 대해서 언급하였으나, 큰 의미가 없는 것으로 보여 생략하였다. 결국 recurrent 계열의 모델보다는 대부분 convolution 계열의 모델에 대해서 많이 사용되고, 연구가 진행되고 있기 때문이다.

Recurrent 모델의 의미는 그래프를 탐색하는 가장 직관적인 방법에서 떠올린 모델로서 의미를 갖는다고 생각한다. 그래프 탐색에서도 DFS가 가장 직관적으로 구현할 수 있는 것과 같은 맥락이다. 이 탐색 과정에서 얻는 결과들이 바로 hidden state가 되고, 이 hidden state 가 모여 결국 그래프의 특성을 결정짓게 되는 방식이다.

하지만 마찬가지로 가장 문제점도 역시 recurrent 부분이 된다. NLP도 결국 recurrent 모델에서 [transformer 모델](/attention-is-all-you-need) 이후로는 대부분 convolution을 사용하고 있다. Recurrent 의 가장 근본적인 문제는 연산이 너무 비효율적이라는 부분에서 시작한다고 생각한다. 이러한 재귀연산의 연산량만큼 그 결과가 가치가 있냐를 봤을 때, 현재까지는 결과적으로 "없다"가 된다. 다른 네트워크로 재귀 없이 병렬계산하여도 결국 좋은 결과들을 얻을 수 있었고, 재귀 방식은 그 재귀의 깊이만큼 의미를 가지지 못했다.

다음 포스팅은 가장 중요하게 이해해야 할 Graph Convolutional Network 모델에 대해서 다뤄본다.