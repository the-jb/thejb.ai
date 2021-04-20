---
layout: post
title: Graph Neural Networks 개념 총정리 3 - Convolutional GNN (1) Spectral-based ConvGNN
tags: [graph, basic knowledge]
---

# 소개

Convolutional Graph Neural Network 는 이전의 Recurrent GNN 과 많이 연관되어있다. Recurrent GNN 에서는 같은 recurrent layer를 계속 반복하며 각 노드의 hidden state 를 일정 step 만큼 업데이트했었는데, Convolutional GNN 에서는 이 recurrent layer 대신에 각 convolutional layer 를 사용한다는 점이 다르다.

여기서 주목해야 할 부분은, Recurrent GNN 에서는 각 step 마다 동일한 레이어를 재귀적으로 사용하는 반면, Convolutional GNN 에서는 각 단계별 convolutional layer를 따로따로 적용한다는 점이다. 이로서 두 가지 다른 점을 생각할 수 있다.

1. **Recurrent Layer 는 단계마다 동일한 가중치를 가진 레이어를 사용하는 반면, Convolutional Layer 는 각 단계별로 다른 가중치를 사용한다는 점이다.**
   이는 얼핏 보면 recurrent 방식이 개념적으로 더 맞지 않나? 생각이 들 수도 있다. Recurrent는 동일한 입장에서 hidden state 의 순환을 바라보는 느낌이 드는데, convolutional 방식은 여러 노드가 다 상황이 다른데, 각 단계별로 다른 가중치를 주는 것의 의미가 있나? 라는 느낌이 들기도 한다.
   하지만 결국 convolutional 방식이 훨씬 더 결과가 좋았고, 이렇게 단계별 가중치가 유용하다는 것을 이해하는 관점에서 Convolutional GNN 을 살펴보면 좋을 것이다.

2. **Recurrent GNN은 `t`번째 레이어를 구하기 위해서는 `t`번의 입력과 출력을 반복하면서 계산을 수행해야 하는 반면 반면, Convolutional GNN은 단순히 `t`개의 Convolutional Layer를 연결하여 한번에 output까지 구할 수 있다.**
   이는 Back propagation 을 할 때, 재귀적으로 가중치를 업데이트 할 필요 없이 한번에 업데이트가 가능하게 되어 연산이 훨씬 효율적으로 수행된다. 따라서 두 방식이 동일한 성능을 내더라도 Convolutional GNN이 훨씬 유용하게 사용될 수 있는 결정적인 이유가 된다.

Convolutional GNN 은 크게 spectral-based 와 spatial-based 의 두 가지 방식으로 나뉘게 된다. spatial-based의 분량이 많아서 포스팅을 나눠서 소개하도록 한다.

# Spectral-based Convolutional Graph Neural Networks

Spectral-based 방식은 그래프 신호 처리 이론을 기반으로 풀어낸 방식이다. Spectral-based 란 스펙트럼에 기반했다는 뜻인데, 여기서 스펙트럼이란 결국 여러가지 파동들의 집합이다. 왜 이러한 이름이 붙었는지 생각해보면, 이 spectral-based 방식에서는 결국 신호처리 이론의 푸리에 변환을 convolution으로 풀어내고 있다. 푸리에 변환이 결국 파동을 분해하는 변환이기 때문에, spectral이라는 이름을 갖게 된 것으로 보인다.

이 spectral-based 방식으로부터 GCN이 탄생하게 되었고, GCN이 좋은 결과를 내면서 이로부터 많은 연구들이 파생되고 있다. 따라서 spectral-based 방식은 GCN을 이해하기 위한 단계로서 접근하는 것이 좋다.

Spectral-based 방식은 기본적으로 무방향 그래프 전제로 하고, 그래프 정보를 normalized graph Laplacian matrix 로 표현하여 사용한다는 것이 특징이다.

## Normalized Graph Laplacian Matrix

Spectral-based 방식에 대해 설명하기 위해서는, 먼저 라플라시안 행렬에 대해 이해하고 있어야 한다. 인접행렬처럼 그래프를 표현하는 하나의 방식이라고 생각하면 되고, spectral-based 방식은 그래프를 이 normalize 된 라플라시안 행렬을 통해 받아들인다.

### Laplacian Matrix

먼저 라플라시안 행렬은 인접행렬에 해당 노드의 차수 정보를 추가한 행렬이다. 차수는 양수값, 인접 여부를 $-1$로 다음과 같이 표현된다.

$$
L^{origin}=D-A
$$

여기서 $A$는 그래프의 인접행렬이고, $D$는 해당 노드의 차수(연결된 엣지 수)를 나타내는 행렬로, $D_{ii}=degree(i), D_{ij(j\neq i)}=0$값의 $n\times n$행렬이다. 결국 $A$와 $D$는 값이 겹치지 않으므로 라플라시안 행렬의 값은 다음과 같다.

$$
L^{origin}_{ii}=degree(i),L_{ij}=-A_{ij}(i\neq j)
$$

이 라플라시안 행렬이 인접행렬의 음수값을 사용하는 이유는, 결국 이 행렬의 각 행이나 열을 합한 값이 0이 되기 위해서이다. 차수만큼의 $-1$값이 존재하기 때문에 각 값들과 차수를 더해서 항상 0이 되는 것이다.

### Normalized Graph Lapliacian Matrix

Normalized graph Laplacian matrix 는 라플라시안 행렬을 normalize 시켜준 행렬이다. 이는 다음과 같은 식으로 표현된다.
$$
L=I-D^{-1/2}AD^{-1/2}
$$

결국 기존 라플라시안 행렬에서 각 차수를 나타내던 $(i,i)$값이 1이 되고, 나머지 인접행렬 값이 차수에 대해서 normalize 시켜준 행렬이 된다. 실제 값은 다음과 같이 들어가게 된다.

$$
L_{ii}=1,L_{ij(i\neq j)}=-A_{ij}/\sqrt{degree(i)degree(j)}
$$

이 normalize 된 라플라시안 행렬은 다음과 같이 분해될 수 있다.
$$
L=U\Lambda U^T,U=[u_0,u_1..u_{n-1}]\in R^{n\times n}
$$
여기서 $U$는 각 고유벡터 $u_i$들을 고유값 크기순서로 정렬한 행렬 나타내고, $\Lambda$는 $\Lambda_{ii}=\lambda_i,\Lambda_{ij(i\neq j)}=0$의 값을 갖는 행렬로, 결국 각 행을 $\lambda_i$배 시켜주는 값이라고 생각하면 된다.

그런데, normalized graph Laplacian matrix 의 고유벡터는 서로 직교한다. 즉 다음과 같은 식이 성립하는 것이다.
$$
U^TU=I
$$
최종적으로 normalized graph Laplacian matrix인 $L$과, 여기에서 얻을 수 있는 $U$가 바로 spectral-based convolutional GNN 을 구하기 위해 사용된다.

## 그래프 푸리에 변환

이제 Normalized graph Laplacian Matrix 를 그래프 푸리에 변환에 사용한다. 여기서 푸리에 변환이 등장하는 이유는, 결국 convolution 연산을 푸리에 변환을 통해서 표현한다고 생각하면 된다. 그래프 신호 $\mathrm{x}\in R^n$를 노드의 각 featㄷure 벡터로 정의할 때, 각 원소 $x_i$는 $i$번째 노드의 값을 나타낸다. 이 경우에 그래프 푸리에 변환을 다음과 같이 정의할 수 있다.
$$
\mathscr{F}(\mathrm{x})=U^T\mathrm{x}
$$
여기서 $\hat{\mathrm{x}}$를 그래프 푸리에 변환을 사용한 결과라고 할 때, 신호 이 함수의 역함수는 다음과 같다.
$$
\mathscr{F}^{-1}(\hat{\mathrm{x}})=U\hat{\mathrm{x}}
$$
그래프 푸리에 변환은 입력 그래프 신호를 normalized graph Laplacian matrix를 통해 고유벡터들의 직교 공간으로 투사된다.  변환된 각 신호 $\hat{\mathrm{x}}$의 원소들은 변환된 공간에서의 위치를 나타낸다. 따라서 입력 신호 $\mathrm{x}$는 다음과 같이 나타낼 수 있다.
$$
\mathrm{x}=\sum_i\hat{x}_iu_i
$$

## Graph Convolution

컨볼루션 필터를 $g\in R^n$라고 할 때, 그래프 컨볼루션 식은 다음과 같다.
$$
\mathrm{x}*_Gg=\mathscr{F}^{-1}(\mathscr{F}(\mathrm{x})\odot\mathscr{F}(g))\\
=U(U^T\mathrm{x}\odot U^Tg)
$$
여기서 $\odot$ 기호는 element-wise 곱셈, 즉 행렬에서 동일 위치에 있는 원소의 곱을 나타내고, $*_G$ 기호는 순환 합성곱인데, 그래프에서의 convolution 정도로 이해하면 

## Sepctral Graph Convolution

여기서 각 convolution filter 를 $g_\theta=diag(U^Tg)$라고 하면, spectral graph convolution 을 다음과 같이 정의할 수 있다.
$$
\mathrm{x}*_Gg_\theta=Ug_\theta U^T\mathrm{x}
$$
Spectral-based Convolutional GNN 은 위의 정의로부터 시작된다. 여기서 필터 $g_\theta$를 어떻게 선택하냐가 중요하다.

### 1. Spectral Convolutional Neural Network (Spectral CNN)

Spectral CNN 은 convolution 필터 $g_\theta$에 대해 다음과 같은 식을 사용한다.
$$
g_\theta=\Theta_{i,j}^{(k)}
$$
이는 여러 채널에서의 학습 파라매터들을 나타낸다. 각 Graph Convolutional Layer 는 다음과 같이 정의된다.
$$
H_{i,j}^{(k)}=\sigma(\sum_{i=1}^{f_{k-1}}U\Theta_{i,j}^{(k)}U^TH_{:,i}^{(k-1)}) (j=1,2,..f_k)
$$
이 식의 구조에 대해서 설명하자면 다음과 같다.

- $k$는 해당 레이어 $H$의 인덱스를 나타낸다.
- $H^{(k-1)\in R^{n\times}f_{k-1}}$은 $H^{(0)}=X$인 그래프 신호에서의 입력이다.
- $f_{k-1}$은 입력 채널이고 $f_k$는 출력 채널이다. 결국 이전 레이어의 출력이 다음 레이어의 입력이 되는 구조가 되는 것이다.
- $\sigma$는 해당 결과에 activation 함수를 적용한다고 생각하면 된다.
- 여기서 필터로 사용된 $\Theta_{i,j}^{(k)}$는 대각행렬이다. 결국 각 대각선의 값이 필터의 weight 가 되는 것이다.

#### Spectral Convolutional Neural Network의 한계점

Spectral Convolutional Neural Network 는 다음과 같은 3가지 한계점이 있다.

1. 그래프에 작은 변화가 있어도 고유벡터들이 변하게 된다.
2. 각 필터들은 domain dependent 하다. 즉 그래프의 구조가 달라지면 적용할 수 없다.
3. 고유값 분해를 계산하기 위한 복잡도가 높다.



이러한 단점을 보완하기 위해 ChebNet, 그리고 그래프에서 널리 활용되는 GCN이 등장하게 된다. GCN은 ChebNet 에서 발전된 이론이다. 따라서 GCN을 이해하기 위해 먼저 ChebNet에 대해 살펴보도록 한다.

### 2. Chebyshev Spectral Convolutional Neural Network (ChebNet)

위에서 필터 $g_\theta$를 단순한 대각행렬로 구성했다면, ChebNet 은 필터를 Chebyshev 다항식으로 다음과 같이 표현한 것이다.
$$
g_\theta=\sum_{i=0}^K\theta_iT_i(\tilde\Lambda)
$$
여기서 $\tilde\Lambda$는 $[-1,1]$ 사이의 값으로, $\tilde\Lambda=2\Lambda/\lambda_{max}-I$이다.

이 필터의 핵심인 Chebyshev 다항식 $T_i(x)$는 다음과 같이 재귀적으로 정의된다.
$$
T_i(x)=2xT_{i-1}(x)-T_{i-2}(x)\\
T_0(x)=1,T_1(x)=x
$$
여기서 $\tilde L=2L/\lambda_{max}-I$를 사용하면, 귀납법으로 $T_i(\tilde L)=UT_i(\tilde\Lambda)U^T$가 되는 것을 증명할 수 있다. 이를 적용하면 최종적인 식은 다음과 같이 된다.
$$
\mathrm{x}*_Gg_\theta=Ug_\theta U^T\mathrm{x}\\
=U(\sum_{i=0}^K\theta_iT_i(\tilde\Lambda))U^T\mathrm{x}\\
=\sum_{i=0}^K\theta_iT_i(\tilde L)\mathrm{x}
$$
위의 Spectral CNN 보다 발전된 점은, 필터들이 Chebyshev 다항식으로 정의되었다는 것이다. 이를 통해 필터들이 local feature 를 독립적으로 추출할 수 있다. 논문에서는 이를 더 일반화시킨 CayleyNet 에 대한 내용도 나오지만, 이는 방향이 다르기 때문에 생략한다.

이 ChebNet은 결국 다항식을 활용하는데, 다항식이라는 것은 우리가 아는 딥러닝과 거리가 멀다. 우리가 알고 있는 딥러닝은 선형 뉴런들이 모여 레이어를 이루고 있기 때문이다. 따라서 이를 우리가 아는 선형의 convolution filter 를 활용하는 방식으로 변형시킨 것이 바로 GCN이다.

### 3. Graph Convolutional Network (GCN)

GCN은 선형 필터를 적용하기 위해 위 ChebNet 을 1차 근사를 적용한 것이다. 위의 식에서 $K=1,\lambda_{max}=2$를 적용하면 다음과 같은 식으로 간소화시킬 수 있다.
$$
\mathrm{x}*_Gg_\theta=Ug_\theta U^T\mathrm{x}\\
=\theta_0\mathrm{x}-\theta_1D^{-1/2}AD^{-1/2}\mathrm{x}
$$
GCN은 이 식에서 over-fitting 을 방지하기 위해 파라매터를 $\theta_0=-\theta_1=\theta$로 만들어 줄였다. 이를 적용하면 위 식을 다음과 같이 쓸 수 있다.
$$
\mathrm{x}*_Gg_\theta=\theta(I+D^{-1/2}AD^{-1/2})\mathrm{x}
$$
이 식이 바로 GCN의 Graph Convolution 식이 된다. 여기에 입력과 출력을 멀티채널의 형태로 만들면 최종적으로 다음과 같은 식이 된다.
$$
H=X*_Gg_\Theta=f(\bar AX\Theta)
$$
여기서 $\bar A=I+D^{-1/2}AD^{-1/2}$이고, $f$는 activaion 함수이다. 그런데 GCN 논문에서는 이 $\bar A$식을 그대로 사용하게 되면 학습의 안정성이 떨어지고, 다음과 같이 normalization 을 활용해서 $\bar A$값을 구하는 것이 더 안정적으로 학습이 진행되어 아래 식을 사용한다고 한다.
$$
\bar A=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}\\
(\tilde A=A+I,\tilde D_{i,i}=\sum_j\tilde A_{i,j})
$$
GCN 을 spatial-based 관점에서도 사용할수도 있는데, 이에 대해서는 다음 포스트에서 다시 다루도록 한다. 이 GCN을 확장한 개념으로 AGCN(Adaptive GCN), DGCN(Dual GCN) 등이 있다.

