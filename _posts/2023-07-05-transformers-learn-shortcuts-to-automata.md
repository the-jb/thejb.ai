---
layout: post
title: "Transformers Learn Shortcuts to Automata"
tags: [논문, NLP, Transformer, Automata]
---

이번 ICLR 2023 에서 정말 멋진 논문을 오랜만에 봐서 이에 대해 포스팅을 남겨보려고 한다.
Transformer 의 동작 방식을 훌륭한 intuition으로부터 증명까지 하고, 이에 대해서 실험적으로도 충분하게 입증한다.
다만, 논문이 상당히 어려운 편이다.
특히 뒷부분 정리의 증명 과정은 사실상 계산이론의 탈을 쓴 추상대수학으로, [Krohn-Rhodes 정리](https://doi.org/10.2307%2F1994127)로부터 세미오토마타를 여러 모듈로 분해하는것부터 출발하는데 이 과정에만 Appendix 30페이지를 넘게 사용한, 거의 이론 저널에 가까운 컨퍼런스 논문이다.

증명에 사용된 계산이론과 추상대수학 이론들은 이러한 이론 연구자가 아니면 완벽히 이해하기 쉽지 않다.
하지만 이 논문은 그러한 증명 디테일까지 다루지 않더라도, 증명 결과로부터 정말 유용한 관점들을 많이 얻을 수 있다.
이 포스팅에서는 증명과정에 대해 심도깊게 다루기보다 이 증명의 결과물들과 아이디어, 그리고 이 것이 어떻게 Transformer를 설명하는지 까지를 추상대수학이나 계산이론에 대한 깊은 지식이 없는 사람들도 이해하기 쉽도록 최대한 자세히 풀어보고자 한다.


- 논문 : [OpenReview](https://openreview.net/forum?id=De4FYqjFueZ), [arXiv](https://arxiv.org/abs/2210.10749)


# 오토마타와 뉴럴 아키텍처

먼저, 제목을 보면 *왜 Transformer 에 오토마타를 학습시켜야 하는가?*, *뉴럴 네트워크에 왜 오토마타가 필요한가?* 이런 질문들부터 떠오를 것이다.
여기서 오토마타는 **우리의 추론 과정을 아주 추상화시킨 형태** 로 이해해야 한다.
즉, 이 논문에서는 Transformer 가 어떻게 추론을 수행하는지, 그리고 아키텍처적으로 근본적인 한계점이 무엇인지까지를 매우 수학적인 형태로 다루고자 하는 것이다.

이 논문에서 다루는 것은 세미오토마타(Semiautomata)로, 일종의 DFA (가장 기본적인 형태의 오토마타) 라고 볼 수 있다. (이보다 확장된 오토마타, 푸쉬다운 등까지는 다루지 않는다.)
그래서 논문을 이해하기 위해서는 가장 먼저 세미오토마타가 무엇인지를 파악해야 한다.


## Semiautomata

![Semiautomata 예제(논문 출처)](.images/transformers-learn-shortcuts-to-automata/semiautomata.svg)

딥러닝에 관심이 있는 사람들이면, 오토마타의 개념보다는 Markov Process가 더 친숙할 것이다.
오토마타는 Markov Process 처럼 어떤 상태가 다른 상태로 전이하는 머신이라고 생각하면 된다.
위 그림과 같이 어떤 상태(state)에서 입력(input)에 따라 다른 상태로 전이하는 과정을 표현한 모델을 세미오토마톤(단수형)이라고 한다.
(왜 semi 라는 표현이 붙었냐면, final state 개념이 없기 때문인데, 이 논문에서는 중요한 부분이 아니니 넘어가도록 한다.)
여기에서 $Q$는 상태의 집합을 말하고, $\Sigma$는 입력의 종류의 집합이다.

위 그림에서 눈여겨봐야 할 부분은 parity counter와 memory unit부분인데, 논문의 뒷부분의 핵심 증명이 되는 단위이다.

Parity bit는 입력의 합이 짝수인지 홀수인지를 구분하는 모듈이다.
이를 더 일반화하면, 입력의 합을 어떤 정수 $n$으로 나누었을 때의 나머지 상태를 알 수 있고, 이를 modular counter라 한다.

메모리 유닛은 회로에서는 Flip-Flop 이라고도 불리는, 1비트를 저장하는 장치를 오토마타로 표현한 것이다.
위 그림에서는 클로버나 다이아 모양을 메모리에서 읽고($\perp$) 쓰는($\sigma$) 작업을 표현하고 있다.
이를 확장하면, $n$비트의 정보를 저장하는 메모리 모듈을 구성할 수 있다.

## 뉴럴 아키텍처

NLP 의 대표적인 아키텍처 RNN과 Transformer는 꽤나 다른 특성을 가지고 있다.
RNN은 거의 일종의 오토마타같이 어떤 state에서 입력을 받으면, 다른 state로 전환하는 재귀 형태의 구조이다.
따라서, 단순하게만 생각해도 위와 같은 세미오토마타를 거의 완벽하게 표현할 수 있다.
여기서 표현한다는 개념은 쉽게 얘기해서 어떤 순차적인 입력들이 들어올 때, 최종적으로 어떤 상태에 도달하는지를 완벽하게 파악하는 모델을 만들 수 있냐는 것이다.

하지만 Transformer는 재귀적인 구조가 아니다.
(물론 autoregressive 하게 동작한다면, 재귀형태를 만들수 있다. 이 형태에 대해서는 아래에서 더 다루도록 한다.)
한 레이어에서 다른 레이어로 순차적으로 전파되기 때문에, 결국 가장 단순하게 생각하면 레이어 개수만큼만 전이가 가능하다라고도 볼 수 있다.
그런데 실제로 Transformer는 레이어 개수에 비해서 훨씬 많은 길이의 문장들을 상당히 잘 처리한다.

이 처리하는 비결이 바로 shortcut을 학습하기 때문이라는 것이 논문 내용의 핵심이다.
그렇다면 도대체 shortcut이란 무엇이고, 이런 결론이 도대체 어떤 의미가 있을까?

# Shortcut

딥러닝을 연구하는 사람들이면 이 shortcut이라는 단어는 매우 친숙하게 보일 것이다.
가장 최근의 LLM 이전까지 전 분야에 걸쳐서 꽤나 핫하게 연구되고, 많은 의문을 제기한 단어이기 때문이다.

바로 [이전 포스팅](/svamp)에서 다루었던 내용도 일종의 shortcut이라고 볼 수 있다.
수학 공식을 정확히 이해했다면, 다른 방식으로 문제를 만들어도 풀 수 있지만, shortcut만을 배운다면 문제를 좀 변형해서 냈을 때 틀린 답을 도출하게 되는 것이다.

이와 같이 shortcut이란 어떤 문제를 해결하는 정해가 아닌 지름길, 즉 *쉬운 길*을 뜻한다.
결국 그 풀이 방식으로는 완벽히 문제를 풀 수 없으며, 어떤 예외적인 경우들이 존재하게 되는 의미를 갖고 있기도 하다.

Transformer가 shortcut을 배운다는 것을 증명하려면, 먼저 shortcut이 무엇인지 정확한 정의가 필요하다.
문제를 푸는 정해가 아니라는 것은 이해하기 쉽다.
하지만 어떤게 *쉬운 길*일까? 이를 어떻게 표현해야 할까?

이 논문에서 오토마타를 사용한 이유는 바로 이런 *길*을 추상화하기 위해서라고 생각한다.
우리가 문제를 추론하는 단계를 오토마타로 표현한다면, 이 오토마타로부터 나타나는 상태변화의 나열하다보면 바로 *길*이 되는 것이다.

그렇다면 다시 돌아가서, shortcut이란 무엇일까?
오토마타로 추상화를 했기에 이제는 쉽게 대답할 수 있다.
바로 원래 정답을 찾기 위해 필요한 상태변화의 길이보다 짧은 길로 정답에 도달하면, shortcut이 되는 것이다.
이를 수학적으로 표현하면 다음과 같다.

> **Definition 1)** *어떤 세미오토마톤 $\mathcal A$에 대해서 $D$깊이의 모델이, 그 깊이보다 더 긴 $T$ 길이 (즉, $o(T)\geq D$) 의 sequence까지 정답을 맞춘다면, 그 모델이 문제를 푸는 방법이 바로 $\mathcal A$에 대한 **shortcut solution**이다.*

정말 직관적이고 유용한 정의가 되었다.
이 논문이 흥미로웠던 이유는 이렇게 실제의 문제들을 정말 직관적으로 와닿게 잘 추상화하고, 그로부터 정말 유의미한 결론들을 얻어냈기 때문이다.

# Transformer Learns Shortcuts

> Transformer는 왜 잘 동작할까? ***Shortcut을 배우기 때문이다.***

이 논문의 목적은 위와 같은 결론을 이끌어 내려는 것이다.
그렇다면 다음과 같은 궁금증이 생길 것이다. Transformer는 (1) 도대체 어떻게 shortcut을 배우고 (2) 그렇게 배운 shortcut은 얼마나 효과적이길래 잘 작동하는 것일까?

먼저 (2)에 대해 살펴보도록 한다.
[Transformer 논문](/attention-is-all-you-need)에서 사용한 레이어어는 깊이 $D=6$을 가진다.
최근의 LLM들은 보통 수십개~100개 까지 레이어를 늘리고 있다.
이런 깊이는 사실 입력 문장의 길이를 생각하면 매우 짧은 편이다.

그렇다면, $D$ 깊이의 레이어는 얼마나 긴 문장까지 소화하는 shortcut을 만들어 낼 수 있을까?
결론부터 말하면 다음과 같다.

> **Theorem 1)** *Transformer 구조는 $O(\log T)$ 만큼의 깊이만 있으면 어떤 세미오토마타든지 $T$ 길이까지 표현할 수 있다. 여기에는 임베딩과 어텐션션의 크기가 상태 개수만큼, $O(\vert Q\vert)$, MLP 크기는 상태 개수의 제곱만큼, $O(\vert Q\vert^2)$ 이 필요하다.*

알고리즘에 익숙한 사람들이라면, $O(\log T)$를 보았을 때 느꼈을 수도 있겠지만, 바로 분할정복 형태의 Transformer 모델을 구성하는 것이다.
아래 그림이 바로 이러한 분할정복 형태의 Transformer를 표현한 것이다.
이제 (1)의 질문에 대답할 수 있다.
각 토큰에마다 모든 상태를 표현할 수 있는 임베딩 크기만 있다면, 입력 크기에 비해 아주 적은 깊이로도 shortcut 동작 모델을 만들어낸 것이다.

![Theorem 1 Intuition (논문 출처)](.images/transformers-learn-shortcuts-to-automata/theorem1.svg)


첫 번째 정리의 핵심은 세미오토마톤을 입력 sequence 길이 $T$ 까지는 $O(\log T)$ 깊이의 모델로 무조건 shortcut을 만들어낼 수 있다는 것에 있다.
깊이가 깊어질수록 습득할 수 있는 문장 길이가 깊이의 지수배만큼 길어지는 것이다.
그래서 재귀 구조가 아닌데도 사실상 현실의 문장들을 훌륭하게 학습해냈던 것이다.

논점과는 다른 방향이긴 하지만, 여기에서 또 한가지 흥미로웠던 부분은, MLP(=Feed-Forward 레이어)의 크기는 다른 레이어들과 다르게 $O(\vert Q\vert^2)$ 이 필요하다는 것이다.
최근 트랜스포머의 Feed-Forward 레이어들의 역할에 대한 연구들을 재미있게 봤는데, 이러한 연구들과 매우 일맥상통하는 결론을 이론적으로 도출하는 점 또한 흥미로운 포인트다.

# Improving Shortcuts

위 분할정복도 상당히 효과적이지만 아쉬운 점이 있다.
Transformer의 어텐션은 global하게 토큰들이 영향을 줄 수 있는 것이 가장 큰 장점인데, 분할정복 형태는 이런 global한 느낌이 없어지기 때문이다.
논문에서는 이러한 직관으로부터 위의 정리를 개선하고자 하였다.
즉, 

*Transformer는 분할정복보다도 더 강력하게 shortcut을 생성할 수 있다.*

라고 얘기하고 싶은 것이다.

이 개선을 위해 이 논문에서 가장 핵심적인 contribution 이자 쉽게 발상하기 어려운 개념인 memory 모듈이 등장한다.
개인적으로, 어떻게 저자가 이런 메모리 형태를 떠올릴 수 있었는지 나로서는 감도 잡히지 않는다.
하지만 결론적으로, 매우 대단한 직관이었고, 이러한 가설을 통해 멋지게 증명을 완성하였다.
또한, 이 가설은 단순히 증명뿐만 아니라 실제 Transformer도 이런 형태로서 동작할 수 있겠구나 라는 직관까지도 주고 있다.

뒤의 메모리 모듈 설명에 앞서 메모리 모듈에 대해 간단하게만 짚고 넘어가면,
먼저 맨위의 그림에서 간단하게 보여주었던 memory unit이 바로 Flip-Flop 메모리 구조이다.
쓰기 입력($\sigma_0$, $\sigma_1$) 이 들어오면 0 혹은 1을 메모리에 쓰면서 그 값을 출력하고, 읽기 입력($\perp$)이 들어오면 메모리에 있는 내용을 출력한다.

이를 좀 더 일반화하면, 어떤 상태 $Q$ 자체를 저장할 수 있는 메모리 유닛을 만들 수 있다.
즉, 여러가지의 쓰기 입력 $\sigma_{q \in Q}$ 과 하나의 읽기 입력 $\perp$을 요구하는 메모리 유닛을 만들 수 있는 것이다.
이를 표현하면 다음과 같다.

> **Memory Unit)** *메모리로부터 읽기 입력($\perp$)이 들어오면, 기억하고 있던 값을 출력하고, 쓰기 입력($\sigma_q$)이 들어오면, 메모리에 있던 값을 $q$로 대체하면서 그 값을 출력하는 모듈이다.*

이제 본격적으로 정리를 이해할 준비가 되었다.
더 강한 shortcut을 만들기 위해서 세미오토마톤을 분해시켜야 한다.

## Krohn-Rhodes Theory

이 분해에 사용되는 이론이 바로 Krohn-Rhodes Theory (1965)이다.
어떤 오토마타와 동일하게 작동하는 다른 형태의 오토마타를 *emulation* 하는 것이다.

여기서 emulation이란, 결과는 동일하지만 그 속의 내부 구조는 전혀 다르게 생겼다는 것을 의미한다.
그러면서도 irreducible, 즉 축소시킬 수 없어야 한다.
무슨 말이냐면, 오토마타에 쓸데없는 장치들을 추가해서 변형시키는 것이 아니라, 아예 근본적으로 다른 원리로 작동해야 한다는 것이다.

완전히 와닿지 않더라도, 이러한 "emulation"을 염두해 두면서 아래 정의를 이해해보도록 한다.
Krohn-Rhodes Theory의 오토마타 버젼은 다음과 같이 표현할 수 있다.
> **Krohn-Rhodes Theory)** *어떤 유한 오토마타 $A$가 주어지면, 이 $A$를 emulate하는 새로운 오토마타 $A'$를 만들 수 있다. 이 오토마타의 형태는 (1) Transformation semigroup을 통해 finite simple group으로 된 오토마타 블럭과 (2) flip-flop 메모리 블럭이 cascade로 연결된 형태로 이루어져 있다.*

메모리 블럭과, 어떤 오토마타 모듈(finite simple group이 될 수 있는)을 엮어서 원래 오토마타를 "emulate"하는 새로운 오토마타를 만들어 낼 수 있다.

원래는 오토마타의 transformation semigroup 과, finite simple group 이 무엇인지에 대한 설명을 추가해 나가고자 하였으나, 엄밀하게 설명하기에는 너무 분량이 많고 어렵기에 이 설명은 생략한다.
간략히 요약하면, transformation semigroup 은 쉽게, 함수의 합성(composition) 개념정도로 이해할 수 있고, finite simple group 은 finite group 의 단위이므로, 결국 finite group 을 만들 수 있고, 이는 결국 어떤 유한한 상태에서 동작하는 오토마타 모듈을 뜻한다.

## Decomposition of Semiautomaton

![Decomposed modules (논문 출처)](.images/transformers-learn-shortcuts-to-automata/decompose.svg)

위의 그림 (b)와 (c)가 바로 이 decomposition으로 만들려는 결과물이다.
먼저, (b)에서 첫 번째 그림이 바로 finite group의 예시를 나타내고, 두 번째 그림이 메모리 모듈이다.
이 논문의 증명의 핵심은 바로 Transformer의 레이어로 위와 같은 두 가지의 모듈을 구현하는 아이디어이다.

### Finite Group

Finite group이란, 유한한 상태에서 작동하는 연산이라고 생각하면 된다..
가장 쉽게 이해할 수 있고, 증명의 바탕이 된 예시가 바로 (b)의 *modular counter* 이다.
상태의 개수를 유한하게 만들기 위해, 나머지 연산을 통해서 쉽게 유한한 상태를 만들 수 있다.

$$
\delta(q, \sigma) = (q+\sigma) \mod n
$$

현재 상태 $q$ 에서 입력 $\sigma$ 만큼 더해서 계속 카운터를 증가시키는 것이다.

이 modular counter는 sum 연산과 mod 연산으로 이루어져 있다.
그리고 transformer는 어텐션 스코어 값을 더하고, 그다음 MLP 레이어를 통과한다.
따라서, 이 MLP 레이어의 weight값을 mod 연산을 수행하도록 weight 를 조절한다면 modular counter를 구현할 수 있다.
(b)의 첫번째 그림이 바로 이 과정을 표현한 모듈이다.

논문에서는 이 modular counter를 기반으로, 일반적인 finite group까지 transformer로 표현할수 있는 것을 증명하였다.

### Memory Unit

Flip-Flop의 개념을 확장하여, 이제 일반적인 메모리 유닛을 만들고자 한다.
Flip-Flop은 1비트만을 저장하거나 불러왔지만, Transformer를 통해 모든 $Q$를 저장하거나 불러오는 모듈을 만들 수 있다.
따라서 입력은, $Q$종류의 쓰기 입력과, 불러오기($\perp$) 종류가 있다. 즉,

$$
\Sigma = Q \cup \lbrace\perp \rbrace, \delta(q, \perp)=q, \delta(q, \sigma\in Q)=\sigma
$$

가 성립하는 모듈을 만들어야 한다.

이는 직관적으로 다음과 같이 생각해 볼 수 있다.
불러오기 입력($\perp$)이 아니면 입력을 그대로 반환하면 되고, 불러오기면 불러오기가 아닌 가장 마지막 입력을 반환하도록 weight를 조절하는 것이다.

Transformer는 입력이 한번에 들어오지만, positional embedding이 결합되기 때문에, 마지막 벡터 여부를 구분할 수 있고, 따라서 이론상 이러한 동작을 하도록 가중치를 설계할 수 있다.

이 과정을 이용하면 (b)의 두번째 그림과 같은 구조가 된다.

### Cascade Product

그림 (c)와 같이, 어떤 레이어 $i$와 그 이후 모든 레이어 $j > i$ 가 연결된 구조를 cascade 구조라고 한다.
Transformer 는 각 레이어 출력을 계산할때 residual connection을 사용하므로, 결국 이러한 cascade 연결구조 형태가 된다.


### Layer Implementation

동일한 단일 Transformer Layer로 위와 같은 Mod counter과 Memory unit 두 종류의 모듈을 구현할 수 있다는 것을 보였다.
그렇다면 구체적으로 두 가지 구현은 어떤 차이가 있을까?

먼저, 가장 주목할 점은 바로 attention의 분포이다.
Modular counter로서 동작하기 위해서는 모든 입력의 합을 구해야 한다.
즉, attention이 균일하게 분포되어있어야 한다는 것이다.
반면 Memory unit은 마지막 벡터를 그대로 반환하므로, 어텐션이 마지막에 치중되어있어야 한다.
따라서, attention이 고르게 분포될수록 mod counter에 가까워지고, 반대로 sparse할수록 memory unit으로 동작하는 형태를 보이는 것이다.

또 한가지는 바로 positional embedding의 영향력이다.
Memory unit으로서 동작하기 위해서는, 실제 토큰의 임베딩값보다 positional embedding을 기준으로 어텐션 참조가 이루어져야 한다.
반대로, mod counter로 동작하려면 positional embedding보다 토큰 임베딩값을 참조해야 한다.

그렇다면 이러한 동작을 활성화시키는 임베딩은 무엇일까?
디테일한 증명과정을 제외하고 직관적인 관점에서 본다면, 먼저 토큰임베딩으로부터 동작의 종류가 구분되어야 할 것이고, 이 종류에 따라 서로 다른 weight를 사용하는 형태가 될 것이다.
Mod counter로 동작하는 weight를 통과하면, 토큰임베딩으로부터 균일한 어텐션이 생성될 것이고, 메모리 유닛으로 동작하는 weight를 통과하면, 포지션 임베딩을 활용하여 sparse attention이 생성될 것이다.


## Theorem 2

Transformer를 통해서 위와 같이 3개의 모듈: finite group(modular counter), memory unit, cascade 를 모두 표현하였다.
따라서 이를 조합하면 다음과 같이 Transformer로 모든 *solvable* 세미오토마타를 만들어낼 수 있다.

> **Theorem 2)** *Transformer는 $O(\vert Q\vert^2 \log \vert Q\vert)$* 로 모든 solvable semiautomata를 표현할 수 있다. 단, 임베딩과 어텐션 크기는 $2^{O(\vert Q\vert \log \vert Q \vert)}$ 가 필요하고, MLP 레이어의 크기는 $\vert Q\vert^{O(2^{\vert Q\vert})} + T\cdot 2^{O(\vert Q\vert \log \vert Q \vert)}$ 가 필요하다.

여기서 solvable이란, 대수학의 solvable group을 만족시키는 것을 의미하는데, 너무 복잡하니 넘어가도록 한다.
어쨌건 Theorem 1보다는 성립하는 세미오토마타의 범위가 줄어들었다는 것만 인지하면 된다.

두 번째 정리의 결론은 입력 길이 $T$가 아무리 늘어도, MLP의 크기만 충분하다면 모든 shortcut을 만들수 있다는 것이다.
그리고, 그 핵심이 되는 방법이 바로 modular counter와 memory unit을 활용하는 것이다.

논문의 Theorem 3은 위 정리에 대한 특수한 케이스, Theorem 4는 sovable이 아닌 인 케이스에 대한 일부 증명으로, 결국, Theorem 2로부터 파생된 케이스들에 대한 내용이기 때문에 생략하도록 한다.

# 어떤 의미가 있는가?

증명 과정은 매우 복잡하지만, 결국 핵심은 transformer가 modular counter와 memory unit 형태를 갖고 있기 때문에 강력한 shortcut 능력이 있다는 것이다.
하지만 이 이론적인 부분은 큰 의미를 갖지 못한다.
왜냐하면, 우리는 사실 transformer를 따로 이러한 모듈로 쪼개기 위해서 학습시키지 않기 때문이다.
다시 말해서, 위의 성능은 그냥 *이론상* 배울수 있는 transformer 구조의 한계치인 것일 뿐이고, 실제 동작과는 전혀 무관한 가설일 뿐이다.

그렇기에 논문에서 가장 흥미로웠던 것이 바로 실험을 통해서 우리가 사용하는 objective로도 실제로 이런 형태의 학습이 일어난다는 것을 보여주었다는 것이다.
논문의 실험에서 다양한 semiautomata 로 만들어진 sequence 예제들을 transformer를 통해 학습시켰고, 이를 요약하면 다음과 같다.

1. Transformer는 이론에서 보여주었던 한계치처럼, 레이어의 깊이보다 훨씬 깊은 sequence까지도 거의 완벽히 학습하고 있었다.
2. 당연하지만, 그 학습결과는 shortcut이고, 더 긴 예제에서는 틀리게 작동한다.
3. 실험에서 정말 놀라운 것은, 이론적인 개념이었던 modular counter와 memory unit형태도 실제로 학습된 weight 속에서 나타나고 있었다.

다시 말해, 위의 증명은 단순히 이론상의 가능성이 아니라, 실제 transformer가 학습하는 방식과 상당히 깊이 관련되어 있는 가설이라는 것을 보여주었다.
이러한 발견을 통해서 이 transformer에 대해 우리는 더 심도있는 직관을 얻을 수 있다.

## 왜 shortcut을 학습하는 것일까?

앞에서 설명했듯이, Transformer 아키텍쳐 자체는 재귀적인 구조가 아니지만, autoregressive하게 입력을 준다면 재귀적인 형태로 transformer가 동작할수도 있다.
그런데 실험 결과는 이러한 셋팅에서도 shortcut으로 학습되고 있었고, OOD 케이스를 올바르게 처리하지 못했다.
왜 이렇게 shortcut 학습 현상이 발생할까?
논문에서는 이러한 이유에 대해서는 명확히 결론내지는 않았지만, 합리적으로 추측해볼만한 충분한 단서들을 제공해 준다.

논문을 읽으면서 느꼈던 가장 핵심적인 차이는 입력방식이었다.

이론상 RNN과 Transformer의 autoregressive decoder는 둘 다 세미오토마톤을 완벽히 표현할 수 있다.

LSTM과 같이 RNN의 경우, 명확하게 기존 state와 입력을 통해 state transition을 발생시키는 방법을 학습한다.
입력을 각각의 timestep별로 따로 학습하기 때문에, 어떤 재귀적인 구조로 학습이 쉽게 발생할수 밖에 없을 것이다.

반면, transformer는 전체 sequence를 하나의 입력으로 간주하고 어떠한 prediction을 수행하도록 동작한다.
재귀 구조의 유무를 떠나서, sequence입력이 한번에 이루어지는 구조에서 학습을 하다보니 objective를 향해 최적화시킬 때, 학습이 정해보다 shortcut으로 유도되기가 더 쉬운 것이다.
그리고 그 결과가 바로 Theorem 2와 같이 메모리와 modular counter형태로 나타났다는 것을 알 수 있다.

논문의 [scratchpad](https://arxiv.org/abs/2112.00114) 실험이 바로 이러한 추측에 대한 단서가 된다.
Scratchpad 학습은 간단히 이야기하면 transformer를 단계별로 따로 학습시키는 방식을 의미한다.
Transformer를 그냥 학습시켰을 때는 shortcut학습이 일어났지만, scratchpad를 통해 입력을 분리하면 autoregressive하게 동작하였을 때의 정해가 학습되었고, OOD에 대해서도 완벽히 처리하는 결과를 보여주었다.

즉, transformer구조의 특성때문에 objective로 수렴하기 위해 정해보다 shortcut으로 먼저 수렴되지만, 이는 충분히 학습방식을 조정하여 해결할 수 있음을 의미한다.
물론 여기에서 주의할 것은, 이러한 shortcut을 활용하지 않고 정해를 학습하는 것이 무조건 좋은 방식을 의미하지는 않는다는 것이다.

NLP의 많은 문제들은 단순히 DFA로 표현할 수 없는 훨씬 복잡한 형태를 띄고 있다.
하지만, 이러한 autoregressive형태는 어떻게 보면 이러한 DFA형태를 강요한다고 볼 수 있고, 반면 shortcut형태는 좀 더 다양한 지름길을 구성할 수 있지 않을까? 라는 생각을 해 볼 수 있다.
물론, 논문 증명에서 사용된 것은 DFA보다 더 적은 solvable semiautomaton에 한정하였기 때문에 이에 대한 근거는 없다.
하지만 DFA를 완벽하게 구현하는 RNN보다 다양한 shortcut을 구현하는 transformer가 실질적인 태스크들에서 훨씬 압도적인 성능을 보여주는것만 보더라도, 이러한 추측은 충분히 합리적이라고 생각한다.


## 논문에서 보여준 것

논문에서 보여준 것은, 어떻게 하면 Transformer 구조가 강력한 shortcut 머신으로 동작할 수 있냐는 것이다.
Theorem 1과 같이 분할정복을 할 수도 있고, Theorem 2처럼 순환구조와 메모리 유닛처럼 동작할 수도 있을 것이다.

Theorem 1은 트랜스포머의 크기에 따른 표현의 한계치를 잘 보여준다.
특히, 깊이가 길어질수록 표현 가능한 shortcut의 길이가 어마하게 늘어나게 되는데, 이 것이 바로 요즘 LLM이 왜 잘 동작하는지를 보여주는 하나의 직관적인 이유라고 볼 수 있다.

Theorem 2는 어텐션 구조의 측면에서 떠오르기 쉽지 않지만 충분히 의미있는 구조를 제시한다.
Global attention을 활용하여 순환 구조와 메모리 유닛을 만들었다.
실질적으로 실험에서도 두 종류의 모듈, flat sum(즉, 순환 구조)과 conditional reset(즉, 메모리 유닛) 형태가 드러나는 어텐션 헤드들이 있는 것을 보여주었고, 이는 정말 의미있는 발견이라고 생각한다.

또, 증명 과정에서 두 Theorem 모두 MLP의 크기가 결국 매우 중요하게 작용하였다.
이는 왜 Feed-Forward 레이어의 중간 히든 사이즈가 커야 하는지를 잘 보여준다.
이와 다른 관점에서 Feed-Forward 레이어가 패턴과 그 패턴에 해당하는 단어를 mapping시키는 역할을 하는 것을 실험적으로 보여준 논문들이 있다.
이 논문들에서 하는 이야기도 결국에는 FF의 크기가 패턴들을 많이 기억하기 위한 메모리의 크기가 되는 것이다.
이 논문에서도 결국 MLP 레이어를 일종의 패턴을 변환하는 용도로 사용하고 있다.
매우 다른 접근이지만, 비슷한 결론에 도달하는 것이다.


## 왜 직관이 중요한가?

이 포스팅에서는 계속 *직관*에 대해서 강조하였다.
도대체 이 이론적인 증명에서 왜 계속 직관만을 얘기하였나? 에 대해서 답하고자 한다.

뉴럴 네트워크는 아직도 블랙박스이다.
트랜스포머 구조가 정말 효과적인 결과를 낸다는 것은 알지만, 대체 어텐션이 구체적으로 어떻게 동작하고 무엇을 학습할 수 있는지는 아무도 정확하게 알지 못한다.
그저 실험적으로 이런 단어들에 어텐션이 많이 가더라 등을 간접적으로 보여줄 뿐이다.

CNN, RNN 등 현재 널리 사용되는 아키텍처들도 결국 직관으로부터 만들어졌다.

특히 자연어 처리에서 Transformer 이후로는 아직까지도 더 나은 아키텍처가 제시되지 않고 있다.
부분적인 개선 방안이 많이 제시되었지만, 결국 큰 의미를 갖지 못하고 LLM과 같이 사이즈를 키우는 것이 전부였다는게 많이 드러났다.

LLM이 되면서 가장 아쉬운 점 중 하나는, LLM은 도대체 왜 강력한 성능을 내나?에 대해서 더이상 분석이 어려워졌다는 것이다.
기존에 Huggingface 등에서 쉽게 공개되었던 다양한 PLM들은 많은 연구자들이 여러가지 관점에서 분석하고 뜯어보면서 그에 대한 의미를 찾았다.
하지만 LLM으로 오면서 모델들은 더이상 많이 공개되지 않았고, 또한 이를 분석하기 위해서도 평범한 장비로는 쉽지 않게 되었다.

그렇기에 이렇게 오토마타를 통해 추상화시키고, 모델 자체의 성능에 대해 이론적인 설명을 해내는 이런 논문이 정말 인상깊었다.
또한, 이를 통해 LLM의 어텐션 안에는 어떤 모듈이 있을지도 짐작해 볼 수 있다.
이런 짐작들을 활용한다면 언젠간 Transformer처럼 shortcut을 배우는 모델이 아닌, 근본적으로 이해하는 모델도 만들어 낼 수 있지 않을까? 생각해본다.

논문에서 사용한 것은 세미오토마타이지만, 이 것은 단순히 오토마타가 아니라 자연어 추론의 가장 추상적인 형태를 나타낸 것이다.
이 논문에서 이런식으로 Transformer의 구조 자체의 이론적 한계 shortcut을 계산하는 접근 방식이 매우 새로웠고, 세미오토마타라는 그래도 꽤 일반적이라고 볼 수 있는 추상 구조에 대해 이렇게 멋지게 증명까지 해낸 점, 또한 그 과정에서 정말 의미있는 intuition들이 많이 등장했다는 점, 이러한 증명이 단순히 이론적인 한계치를 설명한 것이 아니라, 실제 Transformer의 동작방식과 연관이 깊다는 점에서 정말 멋있는 논문이라고 생각한다.
