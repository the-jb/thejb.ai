---
layout: post
title: Attention Is All You Need
date: 2021-03-06 01:45
tags: [nlp, transformer, pytorch]
---

# 자료
- 논문 : [arXiv](https://arxiv.org/abs/1706.03762)

- 구현 코드 : [GitHub](https://github.com/the-jb/attention-is-all-you-need)

# 논문 내용 정리

## 소개

기존 sequence [transduction](/transduction) 모델은 `encoder` 와 `decoder` 구조의 RNN, CNN 을 바탕으로 설계되었다. 이전까지 성능이 가장 좋은 모델들을 보면 `encoder` 와 `decoder` 구조, 그리고 attention 매커니즘을 활용해서 설계되어 있다. 이 논문에서는 **Transformer** 라는 간단하고 새로운 네트워크를 제시한다.
**Transformer** 는 RNN이나 CNN 등 없이 attention 매커니즘 단독으로 기반하여 설계되었고, 놀라운 성능의 결과들을 얻었다.

## 모델 구조

- input sequence of symbol representations = (x1, ... xn)
- continuous representations = (z1, ... zn)
- output sequence = (y1, .. ym)

`encoder` 는 x 를 z 로 매핑한다. `decoder` 는 z를 통해서 y를 생성시킨다.

각 step 에서 모델은 auto-regressive[^1]하고, 이전에 생성된 symbol 을 다음 input에 활용한다.

[^1]: 자기 자신이 입력이자 출력인 모델

아래에서 각 모델을 구성하고 있는 요소를 하나씩 설명한다.

### 1. Encoder 와 Decoder 스택[^3]

- Encoder
  - 인코더는 `N = 6` 개의 동일한 구조의 레이어로 쌓는다.
  - 각 레이어는 다음과 같이 2개의 `sub-layer` 를 가지고 있다.
    - 첫 번째 레이어 : Multi-head self-attention 레이어
    - 두 번째 레이어 : Position-wise fully connected feed-forward
  - 각 sub-layer 마다 residual  connection[^2] 과 normalization 을 적용한다.
  - residual connection 을 편하게 하려고 모든 `sub-layer` 들과 embedding 레이어는 output 차원을 동일하게 `d_model = 512` 로 맞춘다.
- Decoder
  - 디코더 역시 N=6 개의 구조의 동일한 레이어로 쌓는다.
  - 각 레이어는 인코더의 2개 `sub-layer` 에 다음의 3번째 `sub-layer` 를 쌓는다.
    - 세 번째 레이어 : Multi-head attention 레이어 (Encoder 스택의 output 을 반영)
  - 첫 번째 `sub-layer` 인 self-attention 부분은 이후 값이 반영되면 안되므로 Masking 을 적용한다.

[^2]: ResNet 에서의 그 connection, 즉 input 과 output 을 더함
[^3]: 여기서 스택이란, 정말 말 그대로 자료구조의 Stack 이다. Encoder 와 Decoder 를 각각 순차적으로 쌓아 올린 집합을 Encoder 스택, Decoder 스택 이라고 이해하면 된다.
### 2. Attention

Attention 함수는 각 Query에 대해 Key-Value 쌍의 집합이 출력되는 함수이다. Query(Q), Key(K), Value(V), output 은 모두 vector 형태이다.

여기서 Query 는 질문할 단어라고 생각하면 된다. Attention 함수에게 Query 단어에게 영향을 주는 모든 것들에 대해서 질문한다. 그러면 Attention 함수는 각각의 단어(Key)가, Query 에게 얼마나 영향을 주는 지(Value)를 모든 Key에 대해서 `<Key : Value>`쌍의  리스트를 반환한다고 이해하면 된다. 그 것을 보고 각 Key 의 결과물을 Query 에 얼마나 참조할 지 알 수 있는 것이다.

Attention 의 Output 은 각 Value의 가중치 합계를 구한 값이다. 여기서 각 Value의 가중치는 Q, K 로부터 `compatibility function`을 통해 계산된 값이다. `compabitibility function`에 대해서는 아래 Scaled Dot-Product Attention 항목에서 설명된다.

논문에서 Attention 에 다음과 같은 방법들을 적용하였다.

#### Scaled Dot-Product Attention

Attention 함수 중에서는 `compatibility function` 으로 일반적으로 addictive attention 과 dot-product attention 이 가장 많이 쓰인다. 이 논문에서는 기존의 dot-product attention 에 scale 을 적용하여 **Scaled Dot-Product Attention**이라 부른다.

Q, K에 대한 `dimension`을 d<sub>k</sub>, V에 대한 `dimension` 을 d<sub>v</sub>라고 했을 때, dot-product attention 을 수행한 결과를 scaling factor ( 1/sqrt(d<sub>k</sub>) ) 로 나눠준 결과에 softmax 를 적용한 값이 바로 value의 가중치가 된다. 각 value 에 대해서 이 가중치만큼 곱한 결과가 바로 attention 값이 된다. 이를 식으로 표현하면 다음과 같다.

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V (T: 행렬 전치)

Dot-product attention 은 addictive attention 보다 빠르고 효과적으로 학습할 수 있다. 대신 addictive attention 은 d<sub>k</sub>값이 큰 경우에도 효과적인 반면, dot-product attention 은 d<sub>k</sub>값이 커질 경우 결과값이 급격하게 증가할 수 있다. 그렇기 때문에 이 논문에서는 dot-product attention 에 scaling factor를 적용하여 이러한 현상을 방지하고자 한 것이다.

#### Multi-Head Attention

`d_model` 차원의 Q, K, V 단일 attention 함수를 사용하는 대신에 이 논문에서는 Q, K, V 를 선형으로 투사를 h번 하여 이를 학습시키는 것이 효과적이라는 것을 발견했다. 각 Q, K, V 는 d<sub>k</sub>, d<sub>k</sub>, d<sub>v</sub> 의 차원으로 투사된다. 투사된 결과에 d<sub>v</sub>의 output value 가 나오도록 평행해서 차례대로 attention 을 수행한다. 이 결과들을 전부 concat 하여 선형 투사를 하면 최종 결과를 얻을 수 있다. 여러개의 평행한 attention 결과들을 합쳐서 한번에 학습시키기 때문에 이를 **multi-head attention** 이라 부른다. 이를 식으로 표현하면 다음과 같다.

MultiHead(Q, K, V) = concat(`head_1`, head_2, ... `head_h`) * W_O

`head_i` = Attention(Q * WQ_i, K * WK_i, V * WV_i)

여기서 head_i 는 위의 scaled dot-product attention 의 Attention(Q, K, V) 함수를 사용한 결과값을 말한다. 이 논문에서는 `h = 8`개의 평행 attention 레이어를 쌓았고, 각각의 차원 d<sub>k</sub>=d<sub>v</sub>=`d_model / h`=64 를 기본으로 하였다.

#### Multi-head Attention 적용

Transformer 모델은 multi-head attention 을 다음과 같이 3가지 방법으로 사용한다.

1. `encoder`-`decoder` attention

   일반적으로 사용하고 있는 attention 이다. Query 는 이전의 `decoder` 레이어, Key, Value는 `encoder`의 출력값이 된다. 이 것은 decoder 의 모든 결과물들이 input의 각 요소들로부터 얼마나 영향을 받았는지에 대한 attention 이다.

2. `encoder` **self-attention**

   Query, Key, Value 가 모두 `encoder`의 이전 레이어 출력값이 되는 attention이다.

3. `decoder` **self-attention**

   encoder self-attention 과의 차이점은, decoder 의 경우 각 position 에 대해서 해당 position 이전까지만 attention을 적용하기 위해 mask를 사용한다. 그 이유는, 디코더의 컨셉을 RNN처럼 position 별로 순차적으로 출력되는 방향으로 잡았기 때문이다. 이러한 컨셉 하에서 auto-regressive[^1]한 attention 이 되기 위해서는 뒷 position에 대한 정보가 넘어와서는 안된다.

#### Self-Attention

기존의 Attention 은 단순히 input 의 각 요소들이 output의 각 요소들에게 영향을 주는 정도를 계산하기 위한 용도였다. 하지만 이 논문에서는 **self-attention**이라는 기존 attention과는 또 다른 방법의 attention 활용을 적용하고 있다.

self-attention 이란, 말 그대로 자기 자신이 input인 동시에 output이 되는 attention 을 나타낸다. 여기서 자신이란, 하나의 문장 단위라고 이해하면 된다. 하나의 문장 내에서 각 요소(즉, 단어)가 다른 요소들과 얼마나 연관이 있는 지에 대한 정보를 얻기 위해 바로 self-attention 이 사용되는 것이다.

##### self-attention masking에 대해서

사실, 결국 transformer 는 RNN과 다르게 sequence 가 한번에 입력되고 출력된다. 그래서 굳이 `decoder`를 순차적인 모델로 만들어야 할 이유는 없다. 하지만 이 논문에서 `decoder`에만 이러한 컨셉을 잡은 것은, 그 것이 언어 모델에 있어서 더욱 좋은 결과물을 냈기 때문인 것으로 생각된다. 우리가 결국 문장을 만드는 것은, 뒤에 올 단어를 미리 신경써서 만드는 것이 아니라, 보통 앞의 단어들을 이어 나가면서 문장을 만들어 내는 것을 생각해봤을 때 타당하다.

반대로 `encoder` self-attention 이 masking 적용을 안한 이유는, 우리가 문장을 해석하는데 있어서는 문장의 뒷부분을 통해 앞의 단어를 파악하는 등, 단어의 위치에 관계 없이 전체가 다 관계가 얽혀 있기 때문이라고 이해하면 되겠다.

### 3. Position-wise Feed-Forward Networks

각각의 인코더와 디코더 `sub-layer` 는 attention과 더불어 fully connected feed-forward network 도 포함하고 있다. 이 네트워크는 각각의 position 에 개별적으로 동일하게 적용된다. 레이어는 다음과 같은 두개의 선형 함수와 ReLU로 표현된다.

FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b2

다른 포지션에서도 linear transformation 이 동일하게 발생하지만, 레이어마다 다른 파라매터를 사용한다. input과 output 의 convolutional 커널 사이즈는 `d_model`이고, inner layer 는 `d_ff = 2048`이다.

### 4. Embedding

다른 sequence [transduction](/transduction) 모델처럼 학습된 embedding 을 사용하여 `input token` 과 `output token`을 `d_model` 차원을 가진 벡터로 변환한다. 또한 학습된 선형 transformation 과 softmax 함수를 사용하여 `decoder output`을 다음 토큰 확률 예측으로 변환한다. Transformer 모델에서는 두 embedding 레이어와 pre-softmax linear transformation 이 같은 weight matrix 를 공유한다. embedding layer 에서는 이 weight 에 `sqrt(d_model)`을 곱해서 사용한다.

### 5. Positional Encoding

이 모델에서는 RNN이나 Convolution 이 전혀 없기 때문에 모든 입력이 동시에 들어와서 문장에서 토큰의 순서에 대해 전혀 알 수가 없다. 그렇기 때문에 위치에 대해 추가적인 정보 전달이 필요했고, 이 것이 바로 **Positional Encoding**이다. 이 논문에서는 positional encoding 을 `encoder stack`과 `decoder stack`의 시작부분에서 input embedding 에 추가시켰다. 이를 식으로 표현하면 다음과 같다.

PE(pos, 2i) = sin(pos / 10000^(2i / `d_model`))

PE(pos, 2i+1) = cos(pos / 10000^(2i / `d_model`))

여기서 pos는 해당 토큰의 절대 position을 나타내고 2i, 2i+1 은 encoding 결과 matrix에서 해당 차원을 나타낸다. 이렇게 positional encoding 을 하게 되면 결과의 각 차원은 sin파(sinusoid)[^4]가 되며, 파장이 차원마다 2π ~ 20000π 까지 증가하는 형태가 된다.

이 논문에서 이렇게 함수를 사용한 이유는 position의 절대 위치보다 상대위치에 대한 정보를 주고싶었기 때문이다. 예를 들어 어떤 문장의 앞에 k만큼의 문장이 추가되면, 뒤의 문장들은 k 만큼의 offset으로 이동하게 된다. 이는 PE(pos+k) 가 되는데 PE(pos + k) 와 PE(pos)는 항상 같다. 따라서 절대적인 위치에 영향을 받지 않고 상대적인 위치 인코딩이 가능하게 되는 것이다.

[^4]: sin과 cos은 결국 위치 이동을 하면 같아지므로 2i 와 2i+1 둘다 sin파 이다.
