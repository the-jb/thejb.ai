---
title: Transduction
tags: [개념]
---

AI 논문에서 사용하는 transduction 이라는 용어는 다음과 같은 경우들에 따라 전혀 다른 뜻을 갖고 있다. 이는 transduction 이 사용된 주제가 무엇인지를 파악하는 것이 중요하다.

# Transductive Learning

학습 방법론이나 머신 러닝에서의 일반적 해결 방법에 대해서 얘기할 때의 transduction은 induction 과 상반되는 **추론** 방법을 의미한다. 이에 대한 정의는 [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))에 나와있다.

> **transduction** or **transductive inference** is reasoning from observed, specific (training) cases to specific (test) cases. In contrast, **induction** is reasoning from observed training cases to general rules, which are then applied to the test cases.
> 이를 해석하면, **induction**은 training case 에서 일반적인 규칙을 찾아서 test case에 적용을 하는 추론을 하는 반면에 **transduction**은 training, test 케이스 모두 관찰하여 추론이 이루어진다는 뜻이다.

즉, **induction**이 우리가 기존에 알고있는 학습 데이터로 학습을 하고 test case 에 적용하는 방식이다. 반면에 **transduction**은 데이터를 구분하지 않고 관찰해서 추론을 진행한다는 의미를 갖고 있다.

하지만 이러한 설명으로는 와닿지 않고, 막연한 느낌이 든다. 여기에서 여러가지 의문들이 생길 수 있다. 다음에서 이러한 의문들에 대해 살펴보고 해결해 나가면서 **transduction**의 개념에 더 가까이 접근할 수 있다.

## Transductive learning 의 의미 이해하기

다음과 같은 질문들을 통해 **transudction**에 대한 개념을 이해해 보도록 한다.

### 데이터를 학습용으로 사용하면 transduction일까?

딥러닝에서 일반적으로 데이터 파일을 split 하여 일정 비율(예: 70% 학습, 30% 검증)로 사용한다. 위 설명을 보면 그냥 **transduction**을 데이터 100% 학습의 개념으로 오해할 수 있다. 하지만 결론은 "아니다" 이다.

이에 대해 다시 [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))를 참조하면 **transduction**이라는 개념의 도입에 대해서 다음과 같이 설명하고 있다.

> Transduction was introduced by Vladimir Vapnik in the 1990s, motivated by his view that transduction is preferable to induction since, according to him, induction requires solving a more general problem (inferring a function) before solving a more specific problem (computing outputs for new cases): **"When solving a problem of interest, do not solve a more general problem as an intermediate step. Try to get the answer that you really need but not a more general one."**

여기서 굵은 글씨의 문장이 핵심이 된다. 즉, **문제를 풀 때 induction은 일반화를 시키고서 문제에 대한 답을 풀지만, transduction은 일반화 과정이 없이 바로 답을 취한다**는 얘기다.

### 일반화가 없는 딥러닝이란?

위에서 **transduction**은 일반화 과정이 없다고 했다. 그런데 보통 일반화란, 결국 결과물로 어떤 모델을 만들어놓게 된다. 그런데 딥러닝에서 모델을 안만든다는 것이 무슨 의미인지 혼란스럽게 다가올 수 있다.

결론적으로, 딥러닝에서는 이 **transduction**의 일반화 개념을 모델관점이 아닌, 데이터셋 관점에서 해석해야 한다. **Transduction**이 많이 사용되는 그래프 데이터셋을 예로 들면 이해가 쉽다.

나에게 A라는 전체 그래프 데이터셋이 주어지고, 그 A 그래프 안에서만 노드간의 관계등의 여러가지 문제를 풀라고 하면, 바로 **transduction**이다. 나는 주어진 데이터 밖의 다른 그래프(B, C, D..)나 노드들을 예측할 필요가 없다. 주어진 A에 대한 데이터들만 완전히 분석하면 성공적으로 그 문제를 해결했다고 볼 수 있는 것이다.

일반화가 없다는 것은, 바로 A 데이터셋에만 최적화된 모델을 만들면 된다는 의미이다. 나는 먹거리에 대한 그래프 문제만 풀면 되지, 이 모델로 과학 논문에 대한 그래프 문제를 풀 필요가 없다는 것이다.

반대로 여기서 **induction**이란, 그래프 문제를 푸는 어떤 모델을 만들고 학습시켜서, A그래프 문제도 풀고, B그래프 문제도 풀고, C 그래프 문제도 푸는 것이다. 이 것이 바로 위에서 문제를 일반화시켜서 접근한다는 개념인 것이다.

추천 시스템도 대부분 **transduction** 문제라고 할 수 있다. 일반적인 추천모델을 설계할 필요가 없는 것이다. 특정 분야, 상품, 회사에 맞는 고유한 추천모델을 설계하고 풀면 된다. 추천 시스템은 특히 도메인 지식이 많이 활용되기 때문에 어차피 모든 상품들에 대해서 잘 추천해주는 일반화 모델을 만들어내기 어렵다. 결국 우리 회사에서만 잘 돌아가는 추천 모델을 구성하면 되는 것이다.

### 일반적인 딥러닝 모델로 transduction을 푼다면?

위에서는 내 데이터셋에서만 돌아가는 모델을 만들어서 문제를 푼다고 했다. 하지만 많은 논문에서 **transduction**이나 **transductive learning**에 대해서 언급하고 이에 대한 모델을 제시한다. **Transudction**인데, 모델로 여러가지 문제를 푸는 논문에서 의문을 느낄 수가 있다.

여기서의 딥러닝 모델은 단순히 **레이어 구조**가 아닌 **학습까지 완료시킨 모델**로서 이해해야 한다. 동일한 구조의 모델이더라도, 다른 데이터셋에 대해서 학습시키면 결국에 그 데이터셋에 대해서만 풀이하는 모델이 된다면 바로 transduction 모델이 되는 것이다.

딥러닝에서 **transduction**보다 **transductive learning**이라는 표현을 더 많이 쓰는 이유가 바로 이 것이다. 대부분 모델의 구조보다, 그 모델의 학습을 설계하는 관점에서 **transductive** 와 **inductive**가 구분되기 때문이다.

## 결론

**Transduction**을 데이터셋의 관점에서 바라보면 매우 간단하다. 특정 데이터셋 내에서만 문제를 해결하는 모델을 학습시키는 것이 **transductive learning** 인 것이다.

기존에는 레이블로 예시를 들어서 **supervised**와 **unsupervised** 관점과 비교가 필요했다. 하지만 데이터셋 관점에서 보면 간단하게 이를 구분할 수 있다.

- **Supervised/unsupervised learning**은 데이터셋에 label이 존재하는가의 여부로 구분한다.
- **Transductive/inductive learning**은 학습때 사용하는 데이터셋과, 학습된 모델을 활용하는 **데이터셋 자체**의 종류가 같냐, 다르냐로 구분한다.
  즉, 데이터셋 안의 내용은 **transduction/induction**과 전혀 상관이 없는 것이다.

결국 **모델을 학습시키는 목적이 무엇이냐**를 생각해본다면, 딥러닝에서의 **transduction**과 **induction**을 쉽게 구분할 수 있을 것이다.

# (Sequence) Transduction

흔히 자연어 처리에서 seq2seq (sequence-to-sequence) 을 얘기할 때도 **transduction** 용어가 사용한다. 이 경우의 transduction 은 transduce, 변환을 의미한다. 번역 등과 같이 주어진 sequence 를 다른 형태의 sequence 로 변환시키는 태스크에 대한 일반적인 내용을 얘기할 때 주로 사용된다. 많이 사용되는 seq2seq 은 엄밀히 얘기할 때는 sequence transduction 의 한 가지 방법이라고 할 수 있다. 이 때의 seq2seq 은 입력 sequence 의 각 토큰들을 인코더-디코더 구조를 통해서 출력 토큰을 생성해 내는 방법적인 부분을 이야기한다.

결국 대부분의 NLP 의 sequence transduction 문제들이 seq2seq 방식을 사용하기 때문에, 결과적으로 seq2seq 용어를 써도 큰 차이가 없는 경우가 많다. 하지만 [BERT](/bert)나 XLNet 등에서 나오는 auto encoder나 auto regressive 같은 내용들에서는 분명히 둘의 개념에는 차이가 있다. Sequence transduction은 입력 sequence를 다른 형태의 출력 sequence 로 변환시키는 것이다. 따라서 auto encoder 와 같이 입력과 출력 sequence 의 성질이 같은 경우에는 tranduction 이라는 말을 사용하지 않는다.

Transducer 는 어떤 신호를 다른 신호로 변환시키는 도구라는 뜻이다. 따라서 NLP를 제외하고 다른 분야에서도 음성 등 어떤 신호를 다른 형태의 출력물로 변환시킬 때는 transduction 이라는 용어를 사용한다.

