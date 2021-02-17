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

아직 여기까지는 막연한 개념이 된다. 여기에서 다음과 같은 의문들이 생길 수 있다.

# 결국 데이터를 학습용으로 사용하면 transduction인가?

딥러닝에서 일반적으로 데이터 파일을 split 하여 일정 비율(예: 70% 학습, 30% 테스트)로 사용한다. 위 설명을 보면 그냥 **transduction**을 데이터 100% 학습의 개념으로 오해할 수 있다. 하지만 결론은 "아니다" 이다.

이에 대해 다시 [Wiki](https://en.wikipedia.org/wiki/Transduction_(machine_learning))를 참조하면 **transduction**이라는 개념의 도입에 대해서 다음과 같이 설명하고 있다.

> Transduction was introduced by Vladimir Vapnik in the 1990s, motivated by his view that transduction is preferable to induction since, according to him, induction requires solving a more general problem (inferring a function) before solving a more specific problem (computing outputs for new cases): **"When solving a problem of interest, do not solve a more general problem as an intermediate step. Try to get the answer that you really need but not a more general one."**

여기서 굵은 글씨의 문장이 핵심이 된다. 즉, 문제를 풀 때 **induction**은 일반화를 시키고서 문제에 대한 답을 풀지만 하지만, **transduction**은 일반화 과정이 없이, 바로 답을 취한다는 얘기다.

## 딥러닝에서 일반화가 없다는 것은 무슨 뜻인가?

위에서 **transduction**은 일반화 과정이 없다고 했다. 그런데 보통 일반화란, 결국 결과물로 어떤 모델을 만들어놓는다는 의미다. 그런데 딥러닝에서 모델을 안만든다는 것이 무슨 의미인지 혼란스럽게 다가올 수 있다.

결국 여러 transductive learning의 예시들을 찾아보고 내린 결론은, 딥러닝에서는 이 **transduction**의 뜻이 약간 변형되어 사용되는 걸로 보인다는 것이다. 딥러닝에서는 **일반화**라는 개념에 초점을 두기보다, **labeled**와 **unlabeled** 개념에 초점을 두고 있다. 이는 위의 [Wiki에 나온 transduction의 정의](#transduction이란?)부분의 의미와 가깝다. label이 있는 데이터가 바로 정의의 training case를 의미하고, label이 없는 데이터가 test case가 된다.

즉 딥러닝에서 **induction**은 정답이 있는 데이터들을 보고 학습해서 일반적인 규칙을 하고, 정답이 없는 데이터들에 대해서 추론을 한다는 의미를 갖는다.

반면에 **transduction**은 정답이 있는 데이터와 없는 데이터들을 같이 학습시키며 추론하는 개념이다. 정답이 없는 데이터를 어떻게 같이 학습시키냐? 에서 unsupervised learning의 개념을 생각하면 된다. 데이터들을 비슷한 데이터끼리 묶는(clustering) 과정이 추가되는 것이다. 데이터들을 묶어두면, 그 중에 label이 있는 데이터들도 들어있다. 그 label 을 보고 같이 묶여있는 데이터들에 대해서 자연스럽게 label을 추론할 수 있다.

결론적으로 딥러닝에서 일반화의 의미는 학습단계와 추론단계가 분리되어 있을 때, 학습단계를 완료한 모델이 바로 데이터를 일반화시킨 결과라고 생각하면 된다. 반대로 transduction처럼 학습단계와 추론단계가 분리되어있지 않으면, 일반화 단계가 생략된 것이다.

## 그렇다면 transduction에서 예측은 어떻게 하는가?

**induction**에서는 training 데이터로 일반화시킨 모델이 되었다. 그리고 그 완성된 모델로 궁금한 데이터들을 예측한다. 그런데, **transduction**은 무언가 일반화시키는 과정이 없다. 데이터들을 clustering 하고 추론을 진행할 뿐인 것이다. 그렇다면 **transduction**으로 어떻게 예측을 할까?

결론적으로는 **transduction**은 예측모델이라는 개념 자체가 없다고 이해하면 된다. **transduction**은 학습과정이 일반화 없이 label이 없는 데이터들도 같이 추론되기 때문에, 알고싶은 데이터 모두 학습과정에 넣어서 추론을 진행한다. 즉 학습과정 자체가 추론도 포함되는 것이다.

이렇게 할 때 가장 큰 문제점은, 학습시에 없던 새로운 데이터에 대해서 예측하고 싶을 경우다. 다른 특별한 방법을 쓰는 것이 아니면, 결국 새로운 데이터를 예측하려면 다시 모든 데이터들을 가지고 학습 과정을 반복해야 한다.

# 왜 transduction이 필요한가?

여기까지 보면 얼핏 생각하기로, 기존의 **induction**으로 label이 있는 데이터들만 사용해서 모델을 뽑아내는 것이 더 정확하고 깔끔하지 않나? 라고 생각이 들 수 있다. **Transduction**은 어떻게 보면, 굳이 이런 번거로운 과정을 써야 하나 싶을 수 있다.

딥러닝에서 **transduction** 개념이 많이 등장하는 것은, 결국 label이 없는 데이터들이 훨씬 많기 때문이라고 생각한다. label이 확보된 데이터가 많다면 **induction**으로도 충분히 좋은 일반화가 될 수 있을 것이다. 하지만 세상의 대부분의 데이터들은 누군가 label을 달아놓지 않았다. label이 없는 데이터의 수에 비하면 label된 데이터들은 매우 극소수가 된다. 따라서 이런 경우에 **induction**은 극소수의 데이터로만 일반화를 하기 때문에, 오히려 일반화가 이상한 방향으로 진행될 수도 있는 것이다.

딥러닝 이전의 학습 방식은 어느정도 학습모델자체가 그 분야에 맞는 의도된 공식으로 이루어져 있었다. 그렇기에 적은 데이터로 **induction**을 해도 이 의도된 공식 안에서 움직일 수 밖에 없고, 그래서 좋은 예측이 나올 수 있었던 것이다. 하지만 딥러닝은 어떤 분야든 전부 기본적으로 뉴럴넷을 사용하고 학습을 해버린다. 뉴럴넷의 이런 응용력이 높은 만큼, 학습을 잘못하면 이상한 모델이 될 수 있다는 얘기도 된다. 그래서 많은 수의 데이터가 있어야 기본적으로 특이케이스가 적어질 확률이 높고, 좋은 학습이 될 가능성이 높아지는 것이다. 물론 **transduction**의 도입이 딥러닝과 직접적인 관계가 있는 것은 아니지만, 결국 딥러닝에서 **transduction**을 자주 찾는 이유는 결국 이 때문이라고 생각한다.
