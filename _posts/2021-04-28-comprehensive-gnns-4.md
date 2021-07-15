---
layout: post
title: Graph Neural Networks 개념정리 4 - Convolutional GNN (2) Spatial-based ConvGNN
tags: [개념, 논문, Graph]
---

# Spatial-based Convolutional Graph Neural Networks

Spatial-based의 뜻은 공간적인 기반이라는 뜻이다. 공간적인 기반으로 convolution 이 이루어지는 것은 이미지를 예로 들면 쉽게 이해할 수 있다. 이미지에서 CNN은 결국 필터를 통해 중심 픽셀과 근처의 픽셀들을 살펴보기 위한 convolution이 이루어진다. Spatial-based 방식은 그래프에서 노드의 공간적인 관계에 기반해서 convolution을 수행하는 방식이다. 노드의 공간적인 관계란, 결국 가까운 이웃들과 먼 이웃들을 구분한다는 개념으로 이해하면 된다. 결국 중심 노드와 주변 노드들에 대한 정보를 통해 convolution 을 수행하고, 이를 통해 그래프를 분석하는 것이다.

구체적으로 어떻게 spatial-based 가 수행되는지 아래에서 풀어나가도록 한다.

## Neural Network for Graph (NN4G)

NN4G는 최초의 spatial-based ConvGNN 이론으로, GNN 논문에서 같이 제안되었다. NN4G는 노드의 이웃 정보들을 합하여 graph convolution 을 수행하고, residual connection을 적용한다. 결과적으로 다음과 같은 식이 된다.

$$
h_v^{(k)}=f(W^{(k)^T}\mathrm x_v+\sum_{i=1}^{k-1}\sum_{u\in N(v)}\Theta^{(k)^T}h_u^{(k-1)}), h_v^{(0)}=0
$$

이 식을 행렬로 표현하면 다음과 같다.

$$
H^{(k)}=f(XW^{(k)}+\sum_{i=1}^{k-1}AH^{(k-1)}\Theta^{(k)})
$$

여기서 $f$는 위와 마찬가지로 activation function이고, $W^{(k)},\Theta^{(k)}$는 해당 레이어의 파라매터들이다. $A$는 인접 행렬인데, 결국 첫번째 식을 모든 노드에 대해 적용하면 인접행렬을 곱하는 것과 같아진다고 이해하면 되겠다.

## GCN (Spatial-based)

[이전의 spectral-based 포스트](/comprehensive-gnn-3)에서 소개한 GCN 을 spatial-based 식으로 다음과 같이 나타낼 수 있다.
$$
h_v=f(\Theta^T(\sum_{u\in N(v)\cup v}\bar A_{vu}X_u)), \forall v\in V
$$

이를 행렬로 표현하면 다음과 같다.

$$
H^{(k)}=f(\sum_{i=1}^{k-1}\bar AH^{(k-1)}\Theta^{(k)})
$$

NN4G 와의 주요 차이점 중 하나는, NN4G는 그냥 인접행렬 $A$를 사용하는 반면, GCN에서는 위에서 유도했던 $\bar A$를 사용한다는 점이다. 이를 통해 여러 다른 scale의 노드들이 normalize 될 수 있었고, 좋은 결과를 얻게 되었다. 자세한 내용은 [GCN 포스팅](/gcn)에서 확인할 수 있다.

논문에서는 DCNN, DGC, PGC-DGCNN 등 NN4G로부터 개선된 여러 갈래의 논문들의 소개된다. 이러한 논문들이 주로 해결하고자 했던 것은 바로 노드 사이의 거리 문제이다. 노드 사이의 거리가 너무 멀면 해당 노드 사이의 관련이 적어지게 되는데, 이로부터 발생되는 문제들과 해결책을 제시하고 있다. 하지만 이 내용들을 전부 다루지는 않고, 중요한 내용 위주로 소개를 하겠다.

## Partition Graph Convolution (PGC)

그래프가 커질 수록 인접행렬의 크기 또한 많이 증가한다. PGC는 이러한 인접행렬의 크기를 줄여서 계산을 간편하게 하기 위해 노드의 이웃들을 $Q$개의 파티션으로 나눈다. 이 때 최단거리 등을 계산해서 파티션을 나누는 것이 아니라, 특정한 식을 통해서 파티션을 나누게 된다. 그리고 GCN의 식을 적용해서 계산한다. 이를 식으로 표현하면 다음과 같다.

$$
H^{(k)}=\sum_{j=1}^Q\bar A^{(j)}H^{(k-1)}W^{(j,k)}
$$

여기서 $\bar A^{(j)}$는 GCN의 $\bar A$를 각 그룹에 대해서 계산해 준 것으로 이해하면 된다.

## Message Passing Neural Network (MPNN)

MPNN은 어떤 하나의 모델이라기보다, spatial-based ConvGNN 을 일반화시켜 정리한 논문이다.

MPNN에서 가장 중요한 것은 graph convolution연산을 message passing, 즉 노드 사이의 정보 전달로 바라보는 개념이다. 각 노드에는 hidden state인 $h_v$가 있다. 이 hidden state 가 바로 메시지가 되고, 메시지들을 주고받으면서 업데이트가 된다. MPNN은 총 $K$번의 message passing 을 반복해서 정보를 전달하는데, 다음과 같은 과정을 거친다.

먼저 어떤 노드의 hidden state 를 업데이트하기 위해서, 먼저 이웃 노드들의 hidden state 정보들을 모두 전달받아서 통합한다. 이를 식으로 표현하면 다음과 같다.

$$
m_v^{(k)}=\sum_{u\in N(v)}M_k(h_v^{(k-1)},h_u^{(k-1)},x_{vu}^e)
$$

이렇게 받아온 정보 $m_v^{(k)}$를 통해서 해당 노드의 hidden state를 업데이트한다. 이를 식으로 표현하면 다음과 같다.

$$
h_v^{(k)}=U_k(h_v^{(k-1)},m_v^{(k)})
$$

여기서 $U_k,M_k$는 모두 학습 파라매터를 사용하는 정의되지 않은 함수이다. 즉 이 함수들을 자유롭게 구성하여 MPNN 기반으로 학습 모델을 구축할 수 있는 것이다.

최종 레이어 $h_v^{(K)}$는 output layer 로 전달되어, node-level prediction을 수행할 수도 있고, readout 함수를 통해 graph-level prediction을 수행할 수도 있다. 여기서 readout 함수란, 노드의 hidden representation을 통해서 전체 그래프의 representation을 만들어내는 함수이다. 일반적으로 다음과 같이 정의된다.

$$
h_G=R(h_v^{(K)}\vert v\in G)
$$

이 식에서 $R$이 바로 readout 함수가 되는 것이다. 이 또한 위의 $U_k,M_k$처럼 학습파라매터를 사용하는 함수이다. 즉 이 3개의 함수를 정의해서 모델을 구현하면 MPNN 베이스의 모델을 구현할 수 있다.

## Graph Insomorphism Network (GIN)

기존의 MPNN기반의 방식에서 서로 다른 그래프 구조를 그래프 임베딩으로 표현할 수 없다는 단점을 보완하려는 논문이다. GIN에서는 이 문제를 학습파라매터 $\epsilon^{(k)}$를 통해서 중심 노드에 대한 가중치를 조정한다. 이를 식으로 표현하면 다음과 같다.

$$
h_v^{(k)}=MLP((1+\epsilon^{(k)})h_v^{(k-1)}+\sum_{u\in N(v)}h_u^{(k-1)})
$$

여기서 $MLP$는 Multi-layer perceptron, 즉 딥러닝의 일반적인 레이어다.

## Graph Sampling and Aggregate (GraphSAGE)

GraphSAGE는 그래프의 이웃 노드가 너무 많을 경우에도 원활하게 계산을 수행하기 위해 등장했다. GraphSAGE 에서는 이름처럼 샘플링을 통해 참고할 이웃 노드의 개수를 고정시킨다. 이 역시 GCN에 기반하였고, 다음과 같은 식을 통해 구한다.

$$
h_v^{(k)}=\sigma(W^{(k)}\cdot f_k(h_v^{(k-1)},\{h_u^{(k-1)}\vert\forall u\in S_{N(v)} \}))
$$

여기서 $S_{N(v)}$는 $v$의 이웃 노드를 샘플링한 노드들의 집합을 나타낸다. $f_k$는 aggregation 함수인데, 이 함수는 노드의 순서에 영향받지 않는 식으로 정의되어야 한다. 예를 들어서 평균, 총합, 최대값 등이 있다.

## Graph Attention Network (GAT)

Spatial-based 방식들 중에서 가장 핵심적으로 알아야 할 논문이 바로 이 GAT이다. GAT에서는 이웃 노드의 관여를 기여도 관점에서 바라보고 있다. 기여도란, 이웃 노드의 hidden representation 이 해당 노드에 얼마나 영향을 주는지를 나타내는 값이다. 예를 들면 기존의 GCN은 이 기여도를 $\bar A$를 계산해서 사용한다. GraphSAGE 의 경우, 샘플링을 통해서 기여도가 0 혹은 1이 될 것이다.

GCN이나 GraphSAGE 같은 방식은 결국, 미리 기여도를 구해놓고 학습에 사용한다. 하지만 GAT는 이 기여도를 고정시키는 것이 아니라, Transformer 모델의 [attention](/attention-is-all-you-need)을 통해 학습 파라매터로 동작하도록 한다. 구체적인 식은 다음과 같다.

$$
h_v^{(k)}=\sigma(\sum_{u\in N(v) \cup v}\alpha_{vu}^{(k)}\mathrm{W^{(k)}h_u^{(k-1)}})
$$

여기서 $\alpha_{vu}^{(k)}$가 바로 attention 값이 되는 것이다. 주목해야 할 것은, $u$가 이웃노드 뿐만 아니라, 중심 노드 $v$도 포함한다는 점이다. 즉 자기 자신의 hidden representation 값 또한 기여도를 통해서 같이 계산되는 것이다. $\alpha_{vu}^{(k)}$값은 다음과 같은 식을 통해서 계산한다.

$$
\alpha_{vu}^{(k)}=softmax(\mathrm{LeakyReLU(a^T[W^{(k)}h_v^{(k-1)}\vert\vert W^{(k)}h_u^{(k-1)}])})
$$

여기서 $\vert\vert$는 concat 연산자, 즉 행렬들을 나란히 합치는 것을 나타낸다. 이렇게 합쳐진 행렬에 대해 $a^T$를 곱해주는 것이다. 그리고 $a$가 바로 이 attention 의 학습 파라매터가 되며, GAT에서는 activation function 으로 LeakyReLU로 고정하여 정의했다는 것을 알 수 있다. 이를 통해 최종적으로 $\alpha_{vu}^{(k)}$는 $u$의 hidden state들에 대해서 $v$가 영향을 받는 정도에 대한 값을 갖는데, softmax를 사용했기 때문에 이 기여도의 총합은 항상 1이 된다. 

또한 GAT에서는 transformer 의 multi-head attention 또한 적용하였다. GAT 에 대한 자세한 내용은 [Graph Attention Networks 포스팅](/gat)을 참고하면 된다.

결과적으로, GAT는 transformer 모델을 적용하면서 GraphSAGE 에 비해서 node classification 작업에서 놀라운 성능 향상을 보여주었다고 한다. 이에 따라 GAT에서 발전시킨 논문으로 **Gated Attention Network(GaAN)**이 있는데, 이 논문은 transformer 의 self-attention 부분까지 적용시켜주면서 성능을 향상시켰다.

논문에서는 GAT 외에도 많은 spatial-based 논문들에 대해 소개하고 있으나, 결국 우리가 필요한 부분은 그래프에서 주로 다뤄지고 있는 GAT까지만 이해하면 된다고 생각하기 때문에 나머지 소개는 다루지 않는다.

# Spectral-based 와 Spatial-based 모델의 비교

Spectral-based는 기존의 그래프 신호처리 이론으로부터 발전하였고, 여기에서 GCN또한 등장하였다는 점에서 의미를 갖는다. 하지만 결국 현재는 spatial-based 모델이 sepctral-based 보다 더 선호되는데, 그 이유는 다음과 같다.

1. 계산의 효율성

   Spectral-based 모델은 전체 그래프에 대해서 고유벡터를 계산하거나 다뤄야 한다. 하지만 spatial-based 모델은 이웃 노드들에 대해 직접 convolution 연산을 통해서 정보전달을 한다. 이는 그래프의 크기가 커질수록 전체 그래프를 다룰 필요가 없는 spatial-based 모델이 더 계산효율이 좋을 수밖에 없다.

2. 그래프의 변화에 대한 대응성

   Spectral-based 모델은 푸리에 변환에 기반하여 그래프를 다룬다. 따라서 새로운 노드가 추가되는 등, 그래프가 바뀌는 경우는 아예 고유벡터부터 새로 계산해야 해서 모델을 다시 학습해야 한다. 즉 그래프가 고정되어있는 형태에 대해서만 다룰 수 있으며, 그래프의 관계가 변하는 경우를 다루기 어렵다. Spatial-based 모델은 해당하는 노드의 이웃들에 대해서만 새로운 계산을 해주면 된다.

3. 다룰 수 있는 그래프 종류

   Spectral-based 모델은 전제부터 무방향 그래프로 제한되어있다. 하지만 Spatial-based 모델은 얼마든지 방향성 그래프를 다룰 수 있고, 여러 소스의 그래프 입력도 처리가 가능하다.

# 결론

결국 현재 그래프에서 중요한 이론은 spatial-based 에 기반한 이론이다. Spatial-based 모델은 여러 방법들 중에서 GCN을 베이스로 한 많은 연구들이 진행되었고, 그 이후에 GAT를 통해서 노드의 attention을 다루는 것이 주류가 된 것으로 이해하면 된다.