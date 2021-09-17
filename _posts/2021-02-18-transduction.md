---
layout: post
title: Transduction
tags: [개념]
---

Transduction에 대한 포스팅을 이전에 작성했는데, 다시 읽어보니 사족을 너무 많이 붙여서 오히려 개념 이해에 방해가 되는 느낌이었다. Transduction, 혹은 transductive learning이라고 불리는 이 개념은 그렇게 어려운 개념이 아니다. 하지만 이를 번역하는 적절한 한글명이 없어서 처음 개념을 잡을 때 혼란스러울 수 있다. 쓸데없는 설명들을 많이 삭제하고, 개념을 이해할 수 있는 위주로 포스팅을 다시 정리한다.

# Transduction의 정의

**Transduction** 에 대한 설명은 많지만, 정확한 개념 자체를 정의한 것은 거의 찾기 어려웠다. [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))에서만 **transduction**에 대해 다음과 같이 정의되어 있었다.

> **transduction** or **transductive inference** is reasoning from observed, specific (training) cases to specific (test) cases. In contrast, **induction** is reasoning from observed training cases to general rules, which are then applied to the test cases.
> 이를 해석하면, **induction**은 training case 에서 일반적인 규칙을 찾아서 test case에 적용을 하는 추론을 하는 반면에 **transduction**은 training, test 케이스 모두 관찰하여 추론이 이루어진다는 뜻이다.

즉, **induction**이 우리가 기존에 알고있는 학습 데이터로 학습을 하고 test case 에 적용하는 방식이다. 반면에 **transduction**은 데이터를 구분하지 않고 관찰해서 추론을 진행한다는 의미를 갖고 있다.

하지만 이러한 설명으로는 와닿지 않고, 막연한 느낌이 든다. 여기에서 여러가지 의문들이 생길 수 있다. 다음에서 이러한 의문들에 대해 살펴보고 해결해 나가면서 **transduction**의 개념에 더 가까이 접근할 수 있다.

# Transduction 이해하기

다음과 같은 질문들을 통해 **transudction**에 대한 개념을 이해해 보도록 한다.

## 1. 데이터를 학습용으로 사용하면 transduction일까?

딥러닝에서 일반적으로 데이터 파일을 split 하여 일정 비율(예: 70% 학습, 30% 검증)로 사용한다. 위 설명을 보면 그냥 **transduction**을 데이터 100% 학습의 개념으로 오해할 수 있다. 하지만 결론은 "아니다" 이다.

이에 대해 다시 [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))를 참조하면 **transduction**이라는 개념의 도입에 대해서 다음과 같이 설명하고 있다.

> Transduction was introduced by Vladimir Vapnik in the 1990s, motivated by his view that transduction is preferable to induction since, according to him, induction requires solving a more general problem (inferring a function) before solving a more specific problem (computing outputs for new cases): **"When solving a problem of interest, do not solve a more general problem as an intermediate step. Try to get the answer that you really need but not a more general one."**

여기서 굵은 글씨의 문장이 핵심이 된다. 즉, **문제를 풀 때 induction은 일반화를 시키고서 문제에 대한 답을 풀지만, transduction은 일반화 과정이 없이 바로 답을 취한다**는 얘기다.

## 2. 일반화가 없는 딥러닝이란?

위에서 **transduction**은 일반화 과정이 없다고 했다. 그런데 보통 일반화란, 결국 결과물로 어떤 모델을 만들어놓게 된다. 그런데 딥러닝에서 모델을 안만든다는 것이 무슨 의미인지 혼란스럽게 다가올 수 있다.

결론적으로, 딥러닝에서는 이 **transduction**의 일반화 개념을 모델관점이 아닌, 데이터셋 관점에서 해석해야 한다. **Transduction**이 많이 사용되는 그래프 데이터셋을 예로 들면 이해가 쉽다.

나에게 A라는 전체 그래프 데이터셋이 주어지고, 그 A 그래프 안에서만 노드간의 관계등의 여러가지 문제를 풀라고 하면, 바로 **transduction**이다. 나는 주어진 데이터 밖의 다른 그래프(B, C, D..)나 노드들을 예측할 필요가 없다. 주어진 A에 대한 데이터들만 완전히 분석하면 성공적으로 그 문제를 해결했다고 볼 수 있는 것이다.

일반화가 없다는 것은, 바로 A 데이터셋에만 최적화된 모델을 만들면 된다는 의미이다. 나는 먹거리에 대한 그래프 문제만 풀면 되지, 이 모델로 과학 논문에 대한 그래프 문제를 풀 필요가 없다는 것이다.

반대로 여기서 **induction**이란, 그래프 문제를 푸는 어떤 모델을 만들고 학습시켜서, A그래프 문제도 풀고, B그래프 문제도 풀고, C 그래프 문제도 푸는 것이다. 이 것이 바로 위에서 문제를 일반화시켜서 접근한다는 개념인 것이다.

추천 시스템도 대부분 **transduction** 문제라고 할 수 있다. 일반적인 추천모델을 설계할 필요가 없는 것이다. 특정 분야, 상품, 회사에 맞는 고유한 추천모델을 설계하고 풀면 된다. 추천 시스템은 특히 도메인 지식이 많이 활용되기 때문에 어차피 모든 상품들에 대해서 잘 추천해주는 일반화 모델을 만들어내기 어렵다. 결국 우리 회사에서만 잘 돌아가는 추천 모델을 구성하면 되는 것이다.

## 3. 일반적인 딥러닝 모델로 transduction을 푼다면?

위에서는 내 데이터셋에서만 돌아가는 모델을 만들어서 문제를 푼다고 했다. 하지만 많은 논문에서 **transduction**이나 **transductive learning**에 대해서 언급하고 이에 대한 모델을 제시한다. **Transudction**인데, 모델로 여러가지 문제를 푸는 논문에서 의문을 느낄 수가 있다.

여기서의 딥러닝 모델은 단순히 **레이어 구조**가 아닌 **학습까지 완료시킨 모델**로서 이해해야 한다. 동일한 구조의 모델이더라도, 다른 데이터셋에 대해서 학습시키면 결국에 그 데이터셋에 대해서만 풀이하는 모델이 된다면 바로 transduction 모델이 되는 것이다.

딥러닝에서 **transduction**보다 **transductive learning**이라는 표현을 더 많이 쓰는 이유가 바로 이 것이다. 대부분 모델의 구조보다, 그 모델의 학습을 설계하는 관점에서 **transductive** 와 **inductive**가 구분되기 때문이다.

# 결론

이 포스팅의 이전 버젼에서는 너무 모델측면의 설명에 집중한 나머지, 쓸데없이 비약과 사족들을 많이 넣게 되었다. 하지만 애초에 **transduction**을 데이터셋의 관점에서 바라보면 매우 간단하다. 특정 데이터셋 내에서만 문제를 해결하는 모델을 학습시키는 것이 **transductive learning** 인 것이다.

기존에는 레이블로 예시를 들어서 **supervised**와 **unsupervised** 관점과 비교가 필요했다. 하지만 데이터셋 관점에서 보면 간단하게 이를 구분할 수 있다.

- **Supervised/unsupervised learning**은 데이터셋에 label이 존재하는가의 여부로 구분한다.
- **Transductive/inductive learning**은 학습때 사용하는 데이터셋과, 학습된 모델을 활용하는 **데이터셋 자체**의 종류가 같냐, 다르냐로 구분한다.
  즉, 데이터셋 안의 내용은 **transduction/induction**과 전혀 상관이 없는 것이다.

결국 **모델을 학습시키는 목적이 무엇이냐**를 생각해본다면, 딥러닝에서의 **transduction**과 **induction**을 쉽게 구분할 수 있을 것이다.