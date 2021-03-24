---
layout: post
title: Transduction 이란?
tags: [terms, basic knowledge]
---

**Transduction** 에 대응하는 적절한 한글 단어가 존재하지 않는다. 사전을 검색하면 "형질 도입", "변환" 정도의 뜻만 나온다. 왜 딥러닝에서 **transduction** 이라는 단어가 등장하고 사용되는지 나름대로 정리해 보았다.

# Transduction이란?

**Transduction** 에 대한 설명은 많지만, 정확한 개념 자체를 정의한 것은 거의 찾기 어려웠다. [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))에서만 **transduction**에 대해 다음과 같이 정의되어 있었다.

> **transduction** or **transductive inference** is reasoning from observed, specific (training) cases to specific (test) cases. In contrast, **induction** is reasoning from observed training cases to general rules, which are then applied to the test cases.

이를 해석하면, **induction**은 training case 에서 일반적인 규칙을 찾아서 test case에 적용을 하는 추론을 하는 반면에 **transduction**은 training, test 케이스 모두 관찰하여 추론이 이루어진다는 뜻이다.

즉, **induction**이 우리가 기존에 알고있는 학습 데이터로 학습을 하고 test case 에 적용하는 방식이다. 반면에 **transduction**은 데이터를 구분하지 않고 관찰해서 추론을 진행한다는 의미를 갖고 있다.

하지만 이러한 설명으로는 와닿지 않고, 막연한 느낌이 든다. 여기에서 여러가지 의문들이 생길 수 있다. 다음에서 이러한 의문들에 대해 살펴보고 해결해 나가면서 **transduction**의 개념에 더 가까이 접근할 수 있다.

# 결국 데이터를 학습용으로 사용하면 transduction일까?

딥러닝에서 일반적으로 데이터 파일을 split 하여 일정 비율(예: 70% 학습, 30% 검증)로 사용한다. 위 설명을 보면 그냥 **transduction**을 데이터 100% 학습의 개념으로 오해할 수 있다. 하지만 결론은 "아니다" 이다.

이에 대해 다시 [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))를 참조하면 **transduction**이라는 개념의 도입에 대해서 다음과 같이 설명하고 있다.

> Transduction was introduced by Vladimir Vapnik in the 1990s, motivated by his view that transduction is preferable to induction since, according to him, induction requires solving a more general problem (inferring a function) before solving a more specific problem (computing outputs for new cases): **"When solving a problem of interest, do not solve a more general problem as an intermediate step. Try to get the answer that you really need but not a more general one."**

여기서 굵은 글씨의 문장이 핵심이 된다. 즉, **문제를 풀 때 induction은 일반화를 시키고서 문제에 대한 답을 풀지만, transduction은 일반화 과정이 없이 바로 답을 취한다**는 얘기다.

## 딥러닝에서 일반화가 없다는 것의 의미

위에서 **transduction**은 일반화 과정이 없다고 했다. 그런데 보통 일반화란, 결국 결과물로 어떤 모델을 만들어놓게 된다. 그런데 딥러닝에서 모델을 안만든다는 것이 무슨 의미인지 혼란스럽게 다가올 수 있다.

결국 여러 transductive learning의 예시들을 찾아보고 내린 결론은, 딥러닝에서는 이 **transduction**의 뜻이 약간 변형되어 사용된다는 것이다. 딥러닝에서는 **일반화**라는 개념을 조금 다르게 접근해야 한다.

예를 들어 labeled와 unlabeled 데이터가 있다. 위의 [Transduction이란?](#transduction이란)에서 나온 위키 정의부분을 생각해보면, label이 있는 데이터를 training case, label이 없는 데이터를 test case로 생각할 수 있다. 그러면 이 경우에서 **induction**은 정답이 있는 데이터들을 보고 학습해서 일반적인 모델을 만든다. 그리고 그 모델을 통해서 정답이 없는 데이터들에 대해 추론을 하는 것이다.

반면에 **transduction**은 정답이 있는 데이터와 없는 데이터들을 동시에 학습시키며 추론하는 개념이다. 정답이 없는 데이터를 어떻게 같이 학습시키냐? 에서 unsupervised learning의 개념을 생각하면 된다. 데이터들을 비슷한 데이터끼리 묶는(clustering) 과정이 추가되는 것이다. 데이터들을 묶어두면, 그 중에 label이 있는 데이터들도 들어있다. 그 label 을 보고 같이 묶여있는 데이터들에 대해서 자연스럽게 label을 추론할 수 있다.

### 그렇다면 supervised/unsupervised 와 induction/transduction 는 같은 개념인가?

위에서 label에 대해서 이야기를 했다. 그런데 label의 개념을 적용하면, supervised/unsupervised(혹은 semi-supervised) learning 과 개념이 헷갈릴 수 있다. Supervised/unsupervised learning 이 바로 label을 사용하는 지도학습과, label이 없는 비지도학습의 개념이기 때문이다.

하지만 정확히 얘기하면, label의 경우는 하나의 예시로 이해해야 한다고 생각한다. 가장 다른 점이 바로 **일반화**다. 위의 예시를 보면, **induction**은 명확히 학습이 완료되고, 그 모델을 통해서 추론을 한다. 여기서 학습이 완료된 단계가 바로 **일반화된 모델**이라고 얘기할 수 있다. 반대로 **transduction**은 학습과 추론자체가 구분되어 있지 않다. 전체 학습하는 과정 자체가 추론을 진행하는 과정도 같이 포함이 되는 것이다.

정의 자체는 다르지만, 활용면에서 봤을 때는 매우 유사해진다. 결국 **induction**에서 일반화시키기 위해서 웬만해서는 label이 있는 데이터가 필요할수밖에 없기 때문이다. 반대로 정답이 있는 데이터와 없는 데이터를 섞어서 사용하는 semi-supervised learning의 개념들이 **transduction**에서 많이 필요하게 된다.

## 다시 돌아가서, 데이터를 학습용으로 사용하하는 것과 transduction의 차이는?

위에서 데이터를 100% 사용하면 **transduction**인가? 라는 물음에 결론적으로 아니라고 했다. 하지만 위의 글을 읽다보면, 결국 데이터를 전부 사용하면 **transduction**이 되는 것이 아닌가? 라는 생각이 들 수 있다.

이에 대한 답을 위해서 위키의 정의를 다시 살펴보면 아래와 같은 문구로 시작한다.

> **transduction** or **transductive inference** is ...

여기서 **transduction**의 다른 말로 **transductive inference**라고 소개하고 있다. 즉, **transduction**을 단순히 학습의 개념으로 접근하지 말고, **문제를 푸는 전체적인 과정**으로 접근해야 하는 것이다.

문제를 푼다는 것은, 결국 주어진 조건(데이터)으로 예측 결과까지 찾아내는 과정을 의미한다. 그 과정에서 labeled/unlabeled 데이터가 사용될 수도, 데이터를 전부 사용하거나 일부 사용할수도 있다. 하지만 그 문제를 푸는 과정이 일반화(학습)와 예측으로 분리가 된다면 **induction**, 문제 자체를 풀면서 학습을 시킨다면 **transduction**이 된다.

## 그렇다면 딥러닝에서 transduction은 어떻게 예측이 진행될까?

**Induction**에서는 training 데이터로 일반화시킨 모델이 되었다. 그리고 그 완성된 모델로 궁금한 데이터들을 넣으면서 예측한다. 하지만 **transduction**은 무언가 일반화시키는 과정이 없다. 데이터들을 clustering 하며 학습을 진행할 뿐인 것이다. 그렇다면 궁금한 데이터들을 **transduction**으로 어떻게 예측을 할까?

결론적으로 **transduction**은 예측모델이라는 개념 자체가 없다는 것을 다시 한번 이해해야 한다. 딥러닝에서 **transduction**은 학습을 시킬 때, 알고싶은 데이터 모두 학습과정에 넣는다. 결국 모델이 완성되어가면서 동시에 알고싶은 데이터들에 대한 추론들도 같이 진행되는 것이다.

이렇게 진행할 때 가장 큰 문제점은, 학습시에 없던 새로운 데이터에 대해서 예측하고 싶을 경우다. **Induction**에서 일반화 모델을 구하고 나면 완성된 모델을 가지고 계속 예측을 진행할 수 있다. 하지만 **transduction**에서는 다른 특별한 방법을 쓰는 것이 아니면, 결국 새로운 데이터를 예측하려면 다시 모든 데이터들을 가지고 학습 과정을 반복하는 수밖에 없다.

### 모델이 있으니, 거기에 새로운 데이터를 input에만 넣으면 예측이 되지 않을까?

**Transduction**을 일반화과정이 없다고 정의했지만 결국 딥러닝에서는 모델이라는 것이 생성이 된다. 그러면 궁금한 데이터를 넣으면 출력도 되지 않을까? 라고 생각이 들 수 있다. 여기에 대해서는 **어떤 출력은 나오겠지만 좋은 결과가 나오는 것은 아니다**라고밖에 말할 수 없을 것 같다. 그 이유는 결국, 모델이 그렇게 설계되었기 때문이다.

그래프를 예로 들면, 그래프를 transduction으로 푼다면 결국 전체 그래프 구조 데이터들을 다 넣어서 어떤 모델이 완성된다. 그러면서 그 그래프에 대한 문제가 풀린다. 하지만 그 그래프에 어떤 새로운 노드가 추가되었다면, 그래프 구조 자체가 또 달라질 수 있다. 그래서 변형된 그래프 구조를 다 넣어야지 제대로 된 output을 받을 수 있지, 기존의 구조로 학습된 모델로는 제대로 된 output이 나오지 않을 수 있는 것이다.

그냥 개념적인 차원에서 생각해봤을때 내가 내린 답은 위와 같다. 하지만 이에 대해서 더 명확하게 설명하기 위해서는 딥러닝의 여러가지 **transduction**들을 제대로 알아야 할 것 같다.

# 왜 transduction이 필요한가?

여기까지 보면 얼핏 생각하기로, 기존의 **induction**으로 label이 있는 데이터들만 사용해서 모델을 뽑아내는 것이 더 정확하고 깔끔하지 않나? 라고 생각이 들 수 있다. **Transduction**은 문제를 일반화 과정을 생략하고 푼다고 하지만, 어떻게 보면 딥러닝에서는 오히려 복잡한 과정이 추가된 느낌이다. 그래서 굳이 이런 번거로운 과정을 써야 하나 싶을 수 있다.

딥러닝에서 **transduction** 개념이 많이 등장하는 것은, 결국 label이 없는 데이터들이 훨씬 많기 때문이라고 생각한다. 세상의 대부분의 데이터들은 누군가 label을 달아놓지 않았다. 결국 일반적인 **induction**은 극소수의 label된 데이터를 활용해서 일반화를 시키게 된다. 여기서 문제점이 발생한다.

딥러닝 이전의 학습 방식은 어느정도 학습모델자체가 그 분야에 맞는 의도된 공식으로 이루어져 있었다. 그렇기에 적은 데이터로 **induction**을 해도 이 의도된 공식 안에서 움직일 수 밖에 없고, 그래서 좋은 예측이 나올 수 있었던 것이다. 하지만 딥러닝은 어떤 분야든 전부 기본적으로 뉴럴넷을 사용해 학습을 한다. 뉴럴넷이 이렇게 응용력이 높은 만큼, 학습을 잘못하면 이상한 모델이 될 수 있다는 얘기도 된다. 그래서 많은 수의 데이터가 있어야 기본적으로 특이케이스가 적어질 확률이 높고, 좋은 학습이 될 가능성이 높아지는 것이다.

결국 **induction**방식으로 문제를 풀기에는 데이터가 너무 모자라게 되고, 이런 것을 해결하기 위해 label없는 데이터를 잘 활용할 수 있는 **transduction**방식을 찾게 되었다고 생각한다. 물론 위에서 label/unlabel 개념과 induction/transduction의 개념은 다르다고 얘기했다. 하지만 결국 딥러닝에서 **transduction**을 자주 찾고 사용하려는 이유는 여기에 있는 것 같다. 이 데이터 부족을 해결하는 또다른 방향인 few-shot learning도 많은 관심을 받고 있는 것과 같다.