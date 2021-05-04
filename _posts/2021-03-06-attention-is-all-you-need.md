---
layout: post
title: Attention Is All You Need 논문 소개 및 구현
tags: [implement, nlp, transformer, pytorch]
---

# 자료
- 논문 : [arXiv](https://arxiv.org/abs/1706.03762)

- 구현 코드 : [GitHub](https://github.com/the-jb/attention-is-all-you-need)

# 소개

기존 sequence [transduction](/transduction) 모델은 `encoder` 와 `decoder` 구조의 RNN, CNN 을 바탕으로 설계되었다. 이전까지 성능이 가장 좋은 모델들을 보면 `encoder` 와 `decoder` 구조, 그리고 attention 매커니즘을 활용해서 설계되어 있다. 이 논문에서는 **Transformer** 라는 간단하고 새로운 네트워크를 제시한다.
**Transformer** 는 RNN이나 CNN 등 없이 attention 매커니즘 단독으로 기반하여 설계되었고 좋은 성능을 얻어 BERT 등 이에 관련한 여러가지 연구들이 등장하게 되었다.

# 모델 구조

**Transformer 는 크게 `encoder`와 `decoder`를 기반으로 이루어져 있다.**

- input sequence of symbol representations = $(x_1,...x_n)$
  continuous representations = $(z_1,...z_n)$
  output sequence = $(y_1,...y_m)$

  라고 할 때,

- `encoder` 는 $x$ 를 $z$ 로 매핑하는 역할을 한다.
- `decoder` 는 $z$를 통해서 $y$를 생성시키는 역할을 한다.

- 각 step 에서 모델은 *auto-regressive*[^1]하고, 이전에 생성된 symbol 을 다음 input에 활용한다.

[^1]: 자기 자신이 입력이자 출력인 모델

아래에서 이에 대해 하나씩 살펴보도록 한다.

## 1. Encoder 와 Decoder 쌓기

**Encoder 와 Decoder는 각각 같은 레이어를 여러 층으로 쌓은 구조로, 다음과 같이 이루어져 있다.**

- **Encoder**
  - 인코더는 $N(=6)$ 개의 동일한 구조의 레이어로 쌓는다.
  - 각 레이어는 다음과 같이 2개의 `sub-layer` 를 가지고 있다.
    - 첫 번째 `sub-layer` : Multi-head self-attention 레이어
    - 두 번째 `sub-layer` : Position-wise fully connected feed-forward
  - 각 `sub-layer` 마다 *residual  connection*[^2] 과 normalization 을 적용한다.
  - residual connection을 구현하기 편하도록 모든 `sub-layer` 들과 embedding 레이어는 output 차원을 동일하게 $d_{model}(=512)$ 로 맞춘다.
- **Decoder**
  - 디코더 역시 $N(=6)$ 개의 구조의 동일한 레이어로 쌓는다.
  - 각 레이어는 Encoder의 2개 `sub-layer` 구조에 다음의 세 번째 `sub-layer` 를 추가로 쌓는다.
    - 세 번째 `sub-layer` : Multi-head attention 레이어 (Encoder 스택의 output 을 반영)
  - 첫 번째 `sub-layer` 인 self-attention 부분은 이후 값이 반영되면 안되므로 Masking 을 적용한다.

[^2]: ResNet 에서의 그 connection이 맞다. 즉 input 과 output 을 더한 값이 최종 output
## 2. Attention

**Attention은 각 요소에게 다른 요소들이 얼마나 영향력을 줄 수 있는지를 결정하여 최종 output을 도출하는 구조라고 생각하면 된다.**

위에서도 나왔듯이 Encoder와 Deocder 모두 Attention 레이어들을 사용한 구조로 이루어져 있다.

먼저 Attention 함수는 각 Query에 대해 `<Key : Value>` 쌍의 집합이 출력되는 구조이다. Query(Q), Key(K), Value(V), output 은 모두 vector 형태이다.

여기서 Query 는 이해하기 쉽게 관심을 받는 단어라고 생각하면 된다. Attention 함수에게 해당 Query 단어에게 영향을 주는 모든 단어들의 리스트을 요청한다. 그러면 Attention 함수는 각각의 단어(Key)가, 해당 단어(Query) 에게 얼마나 영향을 주는 지(Value)를 모든 Key에 대한 `<Key : Value>`쌍의  리스트를 반환한다고 이해하면 된다. 그 것을 보고 해당 Query에 그 Key의 결과값을 반영할 정도(Value)를 계산하게 된다.

Attention 의 최종 output 은 Attention 함수의 각 Value의 가중치 합계를 구한 값이다. 여기서 각 Value의 가중치는 Q, K 로부터 compatibility function을 통해 계산된 값이다. compabitibility function에 대해서는 아래 Scaled Dot-Product Attention 항목에서 설명된다.

논문에서 Attention 에 다음과 같은 방법들을 적용하였다.

### 2-1. Scaled Dot-Product Attention

Attention 함수 중에서는 `compatibility function` 으로 일반적으로 **addictive attention** 과 **dot-product attention** 이 가장 많이 쓰인다. **이 논문에서는 기존의 dot-product attention 에 scale 을 적용하여 Scaled Dot-Product Attention이라 부른다.**

$Q$, $K$에 대한 `dimension`을 $d_k$, $V$에 대한 `dimension` 을 $d_v$라고 했을 때, dot-product attention 을 수행한 결과를 scaling factor($1/\sqrt{d_k}$ ) 로 나눠준 결과에 $softmax$를 적용한 값이 바로 value의 가중치가 된다. 각 value에 대해서 이 가중치만큼 곱한 결과가 바로 Attention 값이 된다. 이를 식으로 표현하면 다음과 같다.

$$
Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V
$$

#### Dot-Product Attention 과 Addictive Attention 비교

Dot-product attention 은 addictive attention 보다 빠르고 효과적으로 학습할 수 있다. 대신 addictive attention 은 $d_k$값이 큰 경우에도 효과적인 반면, dot-product attention 은 $d_k$값이 커질 경우 결과값이 급격하게 증가할 수 있다. 그렇기 때문에 이 논문에서는 dot-product attention을 사용하며, 그 약점을 보완하기 위해 scaling factor를 적용한 것이다.

### 2-2. Multi-Head Attention

$d_{model}$ 차원의 $Q$, $K$, $V$를 각각의 attention을 사용하는 대신에 이 논문에서는 $Q$, $K$, $V$ 를 선형으로 투사를 $h$번 하여 이를 학습시키는 것이 효과적이라고 한다. 각 $Q$, $K$, $V$ 는 $d_k$, $d_k$, $d_v$ 의 차원으로 투사된다. 투사된 결과에 $d_v$차원의 output value 가 나오도록 평행하게 차례대로 attention 을 수행한다. 이 결과들을 전부 $concat$ 하여 선형 투사를 하면 최종 결과를 얻을 수 있다. **여러개의 평행한 attention 결과들을 합쳐서 한번에 학습시키기 때문에 이를 multi-head attention 이라 부른다.** 이를 식으로 표현하면 다음과 같다.

$$
MultiHead(Q,K,V)=concat(head_1,head_2,...head_h)W^O\\
head_i = Attention(QW^Q_i,KW^K_i,VW^V_i)
$$

여기서 $head_i$는 위 scaled dot-product attention 의 $Attention(Q,K,V)$ 함수를 사용한 결과값을 말한다. 이 논문에서는 $h=8$개의 평행 attention 레이어를 쌓았고, 각각의 차원 $d_k=d_v=d_{model}/h=64$를 기본으로 하였다.

### 2-3. Multi-head Attention의 적용

**Transformer 모델은 multi-head attention 을 다음과 같이 3가지 방식으로 적용한다.**

1. `encoder`-`decoder` attention

   일반적으로 사용하고 있는 attention 이다. Query 는 이전의 `decoder` 레이어, Key, Value는 `encoder`의 출력값이 된다. 이 것은 decoder 의 모든 결과물들이 input의 각 요소들로부터 얼마나 영향을 받았는지에 대한 attention 이다.

2. `encoder` self-attention

   Query, Key, Value 가 모두 `encoder`의 이전 레이어 출력값이 되는 attention이다.

3. `decoder` self-attention

   `encoder` self-attention 과의 차이점은, `decoder` 의 경우 각 position 에 대해서 해당 position 이전까지만 attention을 적용하기 위해 mask를 사용한다. 그 이유는, **디코더의 컨셉을 RNN처럼 position 별로 순차적으로 출력되는 방향으로 잡았기 때문이다.** 이러한 컨셉 하에서 auto-regressive[^1]한 attention 이 되기 위해서는 뒷 position에 대한 정보가 넘어와서는 안된다. 이에 대해서는 아래 [decoder self-attention masking에 대해서](#decoder-self-attention-masking에-대해서)에서 보충 설명한다.

### 2-4 .Self-Attention

**self-attention 이란, 말 그대로 자기 자신이 input인 동시에 output이 되는 attention 을 나타낸다.**

여기서 자신이란, 하나의 문장 단위라고 이해하면 된다. 하나의 문장 내에서 각 요소(즉, 단어)가 다른 요소들과 얼마나 연관이 있는 지에 대한 정보를 얻기 위해 바로 **self-attention** 이 사용되는 것이다.

기존의 Attention 은 단순히 input 의 각 요소들이 output의 각 요소들에게 영향을 주는 정도를 계산하기 위한 용도였다. 하지만 이 논문에서는 **self-attention**이라는 기존 attention과는 또 다른 방법의 attention 활용을 적용하고 있다.

#### 2-4.1 self-attention을 사용하는 이유

이 논문에서는 **self-attention**을 사용하는 이유를 설명하기 위해 **long-range dependency** 에 대해 얘기한다. 여기서 **long-range dependency** 란, 문장에서 요소들 사이의 거리가 증가함에 따른 문장의 요소들간의 의존성을 의미한다. 문장의 길이가 길어지면(즉, 요소들간의 거리가 멀어질수록) 각 요소들간의 연결이 떨어질 수밖에 없는데, 이 연결을 유지시켜주는 목적으로 **self-attention**을 사용한다고 한다.

각 요소들의 연결을 path라 하면, **long-range dependency**를 위해서 이 path의 길이를 줄이는 것이 중요하다. RNN은 sequence가 순차적으로 연결되었기 때문에 $O(n)$이 필요하고, CNN은 $O(log_k(n))$가 걸리는 반면, **self-attention**을 적용하면 모든 요소가 **self-attention**을 통해서 바로 연결되기 때문에 $O(1)$이 된다.

다만, 여기서 레이어의 계산 복잡도도 생각해야 한다. **self-attention**의 레이어 복잡도는 $O(n^2d)$인데, 여기서 문장의 길이 $n$이 너무 길어지면 이 복잡도가 너무 증가하여 계산량이 증가한다. 그래서 문장의 길이가 너무 긴 경우를 위한 **restricted self-attention** 방법도 언급하였다. 일정 길이 $r$만큼 구간을 나누어 **self-attention**을 적용하는 것이다. 이렇게 하면 레이어 복잡도를 $O(rnd)$로 줄일 수 있다. 대신에 path 길이는 $O(n/r)$가 된다.

#### 2-4.2 decoder self-attention masking에 대해서

`decoder`의 input을 자세히 보면, 결국 `decoder` 의 이전 output이 다시 input으로 들어가는 구조가 된다. 이는 실제로는 선형 구조이지만, output을 한번에 뽑아내지 않고 RNN처럼 한 단어의 output을 뽑고, 다시 그 이전 output들과 함께 다음 output을 뽑고 반복하려는 구조이다.

여기서는 다음 두 가지 문제를 살펴봐야 한다.

##### 1. 왜 decoder 를 굳이 순환 output 구조로 만들었나?

사실, 결국 Transformer 는 RNN과 다르게 sequence 가 한번에 입력되고 출력된다. **그래서 굳이 `decoder`를 순차적인 모델로 만들어야 할 이유는 없다.** 하지만 이 논문에서는 **decoder self-attention masking**을 통해서 RNN과 같은 순차적인 모델을 만들려고 하였다.

**이 논문에서 `decoder`에 이러한 컨셉을 잡은 것은, 결과적으로 이 컨셉이 언어 모델에서 더 좋은 결과물을 냈기 때문인 것**으로 생각한다.

이를 사람의 관점에서 개념적으로 생각해보면, **우리가 결국 문장을 만드는 것은, 뒤에 올 단어를 미리 신경써서 만드는 것이 아니라, 보통 앞의 단어들을 이어 나가면서 문장을 만들어 낸다.** 이러한 관점에서 생각해봤을 때, 뒷부분에 대해 masking을 적용시키는 것이 어느정도 타당하다고 느껴진다.

반대로 `encoder` self-attention 이 masking 적용을 안한 이유는, **우리가 문장을 해석하는데 있어서는 문장의 뒷부분을 통해 앞의 단어를 파악하는 등, 단어의 위치에 관계 없이 전체가 다 관계가 얽혀 있기 때문**이라고 이해하면 되겠다.

> **내용 추가**
>
> 이 논문에서는 순환 output구조에서 왜 이전 토큰에 대해서만 참조했는지에 대한 설명이 없었고, 결국 위와 같이 나름의 이유를 추측해서 이해했다. 하지만 [BERT](/bert) 논문에서 이 부분에 대한 명확한 답을 제시해 주었다. 
>
> 결국 위의 얘기가 잘못된 것은 아니었지만, 근본적인 문제는 따로 있었다. 순환 output구조에서도 bidirectional attention을 적용할 수 있었지만, 결국 좋은 방법을 찾지 못했기 때문이었다. 양방향으로 attention을 적용하게 되면, 순환 참조 문제가 발생하게 되고, 이는 overfitting을 유발하게 된다. 또한 타겟을 참조하지 않게 모델을 구상하는 것도 까다롭다.
>
> 그래서 Transformer 모델을 구상할 때는, 1) 한쪽 방향으로만 참조해도 위와 같은 이유에서 충분히 괜찮은 결과가 나오고, 2) 이런 순환 참조와 타겟문제가 자연스럽게 해결되기 때문에 이런 방식으로 모델 구조를 정의한 것이다.
>
> BERT에서는 이런 문제를 해결한 bidirectional 구조를 적용했고, 더 좋은 결과를 얻어냈다. 이에 대한 자세한 내용은 [BERT 포스팅의 해당 부분](/bert#masked-language-model-mlm)에서 확인할 수 있다.

##### 2. masking 을 왜 적용해야 하나?

**위와 같이 순환구조라고 하더라도, masking 이 들어갈 필요는 없다. 왜냐하면 decoder input에 다음 데이터를 넣지 않으면 뒤의 데이터를 참고할 일이 없기 때문이다.** 여기서는 input만 조정하면 이러한 masking이 들어갈 필요가 없다.

하지만 이러한 masking 을 적용한 이유는, 결국 모델부분보다 구현 부분을 살펴봐야 한다.

구현에서는 decoder input에 output sequence를 전체 한번에 입력시킨다. 결국, 선형 처리가 가능한데도 순환 output구조로 학습을 시키는 것 자체가 비효율적이다. 그래서 decoder input에 output sequence 를 한번에 주고, 거기에 decoder self-attention masking을 통해서 뒷부분에 대해 attention을 계산하지 않게 만드는 것이다.

예를 들어 Inputs(encoder input) 가 `"AAAA"`이고, 이를 번역한 결과가 `"BCDE"`라고 할 때, input과 output에 대해서 다음과 같은 차이가 있다.

- 원래대로라면(순환 구조라면)

  | encoder input | decoder input | output |
  | ------------- | ------------- | ------ |
  | "AAAA"        | ""            | "B"    |
  | "AAAA"        | "B"           | "C"    |
  | "AAAA"        | "BC"          | "D"    |
  | "AAAA"        | "BCD"         | "E"    |

  이와 같은 방식으로 한 문장에 4개 데이터를 따로 입력하면 된다. 이런 경우에는 masking을 적용할 필요가 없다.

- 간소화된 구현

  이를 한번에 처리하기 위한 구현으로 논문에서는 다음과 같이 적용한 것이다.
  
  | encoder input | decoder input | output |
| ------------- | ------------- | ------ |
  | "AAAA"        | "BCD"         | "CDE"  |
  
  이와 같이 각 문장에 대해 한번에 처리했기 때문에 뒷부분을 가리는 decoder self-attention masking을 통해서 뒷 단어가 영향을 주지 못하게 만들었다.

## 3. Position-wise Feed-Forward Networks

각각의 인코더와 디코더 `sub-layer` 는 attention과 더불어 fully connected feed-forward network 도 포함하고 있다. 이 네트워크는 각각의 position 에 개별적으로 동일하게 적용된다. 레이어는 다음과 같은 두개의 선형 함수와 $ReLU$로 표현된다.

$$
FFN(x)=ReLU(xW_1+b_1)W_2+b2
$$

다른 포지션에서도 linear transformation 이 동일하게 발생하지만, 레이어마다 다른 파라매터를 사용한다. input과 output 의 convolutional 커널 사이즈는 $d_{model}$이고, 내부 layer 크기 $d_{ff}=2048$이다.

## 4. Embedding

다른 sequence [transduction](/transduction) 모델처럼 학습된 embedding 을 사용하여 `input token` 과 `output token`을 $d_{model}$ 차원을 가진 벡터로 변환한다. 또한 학습된 선형 transformation 과 $softmax$를 사용하여 `decoder output`을 다음 토큰 확률 예측으로 변환한다. Transformer 모델에서는 두 embedding 레이어와 pre-softmax linear transformation 이 같은 weight matrix 를 공유한다. embedding layer 에서는 이 weight 에 $\sqrt{d_{model}}$을 곱해서 사용한다.

## 5. Positional Encoding

이 모델에서는 RNN이나 Convolution 이 전혀 없기 때문에 모든 입력이 동시에 들어와서 문장에서 토큰의 순서에 대해 전혀 알 수가 없다. 그렇기 때문에 **위치에 대해 추가적인 정보 전달이 필요했고, 이 것이 바로 Positional Encoding이다.** 이 논문에서는 positional encoding 을 첫 `encoder`과 `decoder`의 시작부분에서 input embedding 에 추가시켰다. 이를 식으로 표현하면 다음과 같다.

$$
PE(pos,2i)=sin(pos/10000^{2i/d_{model}})\\
PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}})
$$

여기서 $pos$는 해당 토큰의 절대 position을 나타내고 $2i$, $2i+1$ 은 encoding 결과 matrix에서 해당 차원을 나타낸다. 이렇게 positional encoding 을 하게 되면 결과의 각 차원은 *사인파(sinusoid)*[^4]가 되며, 파장이 차원마다 $2\pi$ ~ $20000\pi$ 까지 증가하는 형태가 된다.

이 논문에서 이렇게 함수를 사용한 이유는 **position의 절대 위치보다 상대위치에 대한 정보를 주고싶었기 때문이다.** 예를 들어 어떤 문장의 앞에 k만큼의 문장이 추가되면, 뒤의 문장들은 k 만큼의 offset으로 이동하게 되는데, 이는 식으로 $PE(pos+k)$ 가 다. **그런데 $PE(pos + k)$ 와 $PE(pos)$는 항상 같다.** 따라서 **절대적인 위치에 영향을 받지 않고 상대적인 위치에 대해서 인코딩하여 값을 넘겨줄 수 있다.**

[^4]: sin과 cos은 결국 위치 이동을 하면 같아지므로 2i 와 2i+1 둘다 sin파 이다.

# 학습 조건

## 정규화

이 논문에서는 다음과 같이 2가지 정규화 요소를 사용했다.

- Dropout

  이 논문에서 dropout 을 크게 두 가지 부분에 적용하였다.

  - 첫 번째로, 각 sub-layer의 output부분에 residual & normalization 을 하기 전에 dropout을 먼저 적용하고 residual connection 을 수행했다.

  - 두 번째로는 각 encoder와 decoder 에서 positional encoding 을 추가한 이후에 encoder input으로 들어가기 직전에 dropout을 수행했다.

  논문에서 dropout 비율 $P_{drop}=0.1$로 설정했다.

- Label Smoothing

  논문에서는 $\epsilon_{ls}=0.1$의 label smoothing을 적용하였다. BLEU 점수에 도움이 되었다고 한다.

## 데이터

논문에서는 WMT2014 의 영어-독일어, 영어-프랑스어 데이터를 사용한 부분에 대해서만 나와있다.

## Optimizer

논문에서는 Adam Optimizer를 $\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$로 적용하였다.

learning rate 는 다음과 같이 적용하였다.

$$
lrate=d_{model}^{-0.5}min(step^{-0.5},step\times warmup\_steps^{-1.5})
$$
여기서 $warmup\_steps=4000$을 사용했다.

