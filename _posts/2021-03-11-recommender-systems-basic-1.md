---
layout: post
title: Recommender Systems의 기초 (1)
tags: [recommender system]
---

# 참고 논문 및 자료

- G. Adomavicius and A. Tuzhilin, "Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions"

# Recommendation Problem

## 1. Recommandation Problem 이란?
추천 시스템은 **Collaborative Filtering** 을 필두로 하여 연구가 본격화되고, 다양한 분야에서 다양한 형태로 사용되고 발전하고 있다. 추천 시스템을  **Recommandation Problem**으로 정의하고 이 문제를 해결해 나가는 방식에 대해 다룬다.

- 가장 일반적으로 Recommendation Problem 은 아이템에 대한 유저의 `rating`을 추정하는 것이라고 할 수 있다.
- 이 `rating` 을 추정하는데 가장 쉽게 떠올릴 수 있는 데이터는
  1. 그 유저의 다른 아이템에 대한 `rating`값
  2. 다른 유저의 그 아이템에 대한 `rating`값이다.

## 2. 수식을 통해 Recommandation Problem 구체화
Recommendation Problem 을 식으로 구체화하여 정의하면 다음과 같다.

- 모든 c에 대해 **s' = argmax u(c, s)**
  (**s'** : 유저에게 추천할 아이템, **c** : 유저, **s** : 아이템, **u** : utility function, c에게 s가 얼마나 유용한지에 대한 함수)
- 유저 공간을 나타내는 **C**는 그 유저의 성별, 나이 등의 profile 같은 부분으로 구체화할 수 있다.
- 아이템 공간의 **S**도 마찬가지로 그 아이템의 속성들을 구체화할 수 있다.
- **u**도 보통 모든 **C×S** 공간에 대해 알고있지 않고, 그 일부분에 대해서만 정의되어 있다. 따라서 모르는 부분에 대해서 u 함수를 `extrapolate`[^1] 해야 한다.

u 함수에 대해 extrapolate 하는 것은 주로 1) 휴리스틱하게 함수를 설계하거나, 2) 추정한 결과를 특정 performance criterion[^2] 을 통해 최적화시키는 단계로 진행된다. 이런 과정을 통해 `rating` 값들이 추정이 되면, 위 **s'**에 대한 식을 통해 유저에게 아이템을 추천하는 것이 바로 Recommendation Problem 을 푼다고 할 수 있다.

[^1]: 두 점 바깥의 값을 예측. 여기서는 이전 데이터들로 앞으로의 데이터를 추측한다는 의미로 보면 됨 (참고. interpolate : 두 점 사이의 값을 예측)
[^2]: 함수의 performance 를 추정할 수 있는 식. 예를 들어서 MSE(Mean Square Error) 
## 3. Recommandation Problem 해결 방법의 종류

결국 아직 정해지지 않은 `rating`값을 추측하는 방법이 바로 핵심이 된다. 이에 대해서는 다음과 같은 방법들이 가장 기본적인 접근 방법이다.

- Content-based recommendation
  유저가 이전에 좋아했던 '아이템' 을 기반으로 비슷한 아이템을 추천
- Collaborative recommendation
  그 유저와 비슷한 성향의 '사람'들이 좋아하는 아이템을 추천
- Hybrid approaches
  두 가지 방식을 혼합하여 사용

이에 대해서 아래에서 자세히 다뤄 본다.

# Content-Based Recommendation

- Content-based 방법이란 u(c, s)를 구하기 위해 이미 알고있는 u(c, s<sub>i</sub>)으로부터 추정하는 방법을 의미한다.
  - 여기서 s<sub>i</sub> 란 s와 "비슷한" 아이템을 나타내고, 같은 c를 고정시키고 다른 아이템 s<sub>i</sub>에 대한 utility 값으로부터 추정을 하는 것이 특징이다.
- Content-based 접근은 **Information Retrieval** 로부터 발전한 결과이다.
  - **Information Retrieval**은 여러 웹사이트, 문서 등에 대한 검색으로부터 출발했기 때문에 text-based 컨텐츠인 경우가 많다.
  - **Information Retrieval**보다 발전된 점은 바로 **user profile** 을 사용한다는 점이다. user profile 에는 유저의 취향이나 니즈 등의 정보를 담고 있다.

아이템 s의 `item profile` 을 Content(s) 라고 할 때, 이 `item profile`은 아이템 s의 컨텐츠로부터 feature 들을 추출하여 계산된다.

Text-Based 컨텐츠에서 보통 중요한 feature 는 바로 **키워드**이다. 이 키워드를 뽑아내는 대표적인 방법이 **Term Frequency - Inverse Document Frequency (TF-IDF)** 이다.

## Term Frequency - Inverse Document Frequency (TF-IDF)

Text-based 컨텐츠에서는 유사한 아이템을 찾기 위해 그 컨텐츠의 주요 키워드를  찾아 비교하기도 한다. **TF-IDF** 는 text-based contents 에서 각 키워드의 중요성을 구하기 위한 가장 기본적인 측정 방식이다.

- **TF-IDF** 는 문서 안에서의 키워드의 빈도수 TF 와 특정 키워드가 등장한 문서 개수 DF를 이용하여 다음과 같이 구한다.

  - **Term Frequency(TF)**는 특정 문서에서 각 키워드가 얼마나 자주 등장하냐를 측정한다.
    f(i, j) 는 특정 키워드 k<sub>i</sub>가 문서 d<sub>j</sub> 에서 가지는 빈도수(frequency) 라고 할 때,

    > TF(i, j) = f(i, j) / max(z, f(z, j))

    따라서 TF가 높을수록 문서와 키워드의 연관성이 깊다고 할 수 있다.

  - **Document Frequency(DF)**는 전체 문서에서 해당 키워드가 등장한 문서가 얼마나 많냐를 측정한다.

    전체 N개의 문서 중에서 키워드 k<sub>i</sub>가 등장한 문서의 개수를 n<sub>i</sub>라고 할 때,

    > DF(i) = n(i) / N

    따라서 DF 가 높을수록 그 키워드는 어떤 문서든지 자주 등장하기 때문에 키워드로서의 중요성이 낮아진다고 할 수 있다.

  - TF 가 높을수록, DF 가 낮을수록 키워드의 중요성(weight)의 값은 증가한다.

    따라서 **Inverse Document Frequency(IDF)**는 `IDF(i) = log(1 / DF(i))` 가 된다. (비율을 일정하게 적용하기 위해 역수에 log 적용)

  - 결국, 전체 **TF-IDF** weight 값은 다음과 같다.
    
    > w(i, j) = TF(i, j) * IDF(i)

- 이렇게 구한 **TF-IDF**값을 통해 content 의 keyword의 weight 를 구해서 주요 keyword를 뽑아낼 수 있다.

따라서 TF-IDF 를 통해 특정 문서 d<sub>j</sub>의 content 를 다음과 같이 정의할 수 있다.

> Content(d<sub>j</sub>) = (w1_j, w2_j, ... wk_j, ... wK_j) (wk_j : 문서 d<sub>j</sub>의 k번째 키워드, K : 키워드의 개수)

## Content-based Profile

**TF-IDF** 로 item 의 유사 contents 를 구할 수 있었다. 이것을 바탕으로 **Content-based Profile** 을 구할 수 있다.

- **Content-based Profile** 은 유저의 취향 정보를 담고있는 `profile`이다. 이 `profile`은 이전에 유저가 경험하고 `rating`을 매겼던 아이템을 바탕으로 작성된다. 
- ContentsBasedProfile(c) = (w1, w2, ... wk) (w : 각 keyword가 유저에게 중요한 정도의 weight 값) 을 이용해 rate 를 계산할 수 있다.
  - Rocchio 알고리즘으로 average 값을 구하는 방법이 있다.
  - Bayesian classifier 를 통해 해당 document 의 확률을 추측할 수 있다.
  - Winnow 알고리즘도 feature 가 많을 경우 효과적으로 사용할 수 있다.

이 ContentBasedProfile(c) 를 통해 u(c, s)를 다음과 같이 정의할 수 있다.

> u(c, s) = score(ContentBasedProfile(c), Content(s))

위에서 언급된 Information Retrieval 을 바탕으로 한 컨텐츠에서는 ConentBasedProfile(c) 와 Content(s) 둘 다 TF-IDF 벡터(각각 w<sub>c</sub>, w<sub>s</sub>)로 표현될 수 있다. 그리고 이제 score 함수를 구하는 방법들이 필요하다.

1. **휴리스틱 함수로 적용하는 방법**

   score 를 휴리스틱한 식으로 구하여 사용할 수 있다. 대표적인 휴리스틱 함수가 바로 코사인 유사도 함수이다. 코사인 유사도 함수를 이용해 score 를 다음과 같이 구할 수 있다.

   > u(c, s) = cos(w<sub>c</sub> 벡터, w<sub>s</sub> 벡터)
   > = sum(w_c(i) * w_s(i)) / sqrt(sum(w_c(i)^2) * sum(w_s(i)^2)) (i : 각 키워드 i)

   이 식을 풀이하면, 결국 유저에게 중요한 키워드 k가 그 문서에서 중요한 키워드 일수록 그 문서에 대한 u(c, s) 값이 높아지는 것이다.

2. **러닝 기법들을 통한 모델을 적용하는 방법**

   기존의 휴리스틱한 식으로 구하는 대신에, 머신러닝이나 통계적 학습 등을 통해 모델을 구하고, 이를 통해 score 를 추측하는 방법들이 있다.

   Web page 에서는 유저에게 <"관련있는", "관련없는"> 두 개의 항목으로 분류할 수 있는데, 이렇게 분류하는 방법으로 **Bayesian classifier(베이지안 분류)** 가 있다.

### Bayesian classifier

베이지안 분류는 문서(웹 페이지)를 여러 클래스 C<sub>i</sub>로 분류하기 위해 사용되는 방법이다. 특정 키워드 k<sub>j</sub>가 있는 상황에서 그 문서가 C<sub>i</sub>클래스일 확률을 구해야 한다.

> P(C<sub>i</sub> | k<sub>1</sub> ~ k<sub>n</sub>)

여기서 키워드들이 서로 독립적이라면, 위 확률은 아래와 같은 식으로 구할 수 있다.

> P(C<sub>i</sub> | k<sub>1</sub> ~ k<sub>n</sub>) = P(C<sub>i</sub>) × Π P(k<sub>x</sub>|C<sub>i</sub>)

이러한 식으로 바꾼 이유는 P(C<sub>i</sub>)와 P(k<sub>x</sub>|C<sub>i</sub>)는 학습 데이터에서 쉽게 구할 수 있기 때문이다. 또한, 실제로 키워드들은 서로 완전히 독립적일 수는 없는데, 키워드들이 독립적이지 않아도 좋은 정확도를 가질 수 있다.

### Content-based 추천 방식의 한계

위와 같은 방법들 이외에도 많은 다양한 방법들이 있다. 하지만 content-based 추천 시스템 내에서 적용할 수 있는 방법들은 한계가 있다. 그 한계들은 다음과 같다.

1. **content-based 방식을 적용할 수 있는 content가 많지 않음**

   content 를 분석할 때, 아이템과 명확하게 상관관계가 있는 feature만 분석이 가능하다. 따라서 그런 feature 를 추출할 수 있는 content 들에만 이 방식을 적용할 수 있다.

   또한, 이렇게 추출한 feature 가 같은 경우, 해당 아이템을 구분할 수 없다는 것도 문제가 된다. 따라서 중복되기 어려울 정도로 자세한 feature 를 추출할 수 있는 content 여야 한다.

   위에서 언급한 text retrieval 등에는 유용하게 사용할 수 있지만, 이런 것들이 적용되기 어려운 content들은 이 추천 방식을 사용할 수가 없다.

2. **Overspecialization**

   유저는 과거의 자신의 선택에 대해서 연관된 것들만 추천을 받는다. 따라서 추천을 통해 과거와 다른 새로운 content를 경험해볼 수 없게 된다. 또한, 반대로 비슷한 컨텐츠를 찾다보니, 이미 경험한 컨텐츠를 계속 중복 추천해주는 문제도 생기게 된다.

   이러한 문제를 overspecialization(과잉 전문화) 라고 한다. 결국 content-based 추천의 전제 조건은 **유저가 어떤 아이템을 좋아하면, 그 아이템과 비슷한 아이템도 좋아한다**가 되어 이를 기반으로 utility function 을 구하는 것이다. 하지만 overspecialization 문제는 이 조건 자체가 옳지 않다는 것을 뜻한다. 유저는 안 비슷한 아이템을 좋아할 수도 있고, 반대로 비슷한 아이템도 싫어할 수가 있다.

3. **새로운 유저는 정보가 없어서 추천이 어려움**

   유저의 과거 컨텐츠에 기반해서 추천을 해주다보니, 과거 컨텐츠 기록이 적은 신규 유저에게는 좋은 추천이 가기가 어렵게 되는 문제가 있다.

# Collaborative Recommendation

Collaborative filtering 이라고도 불리우는 Colaborative recommendation 은 위에서 언급한 바와 같이, 다른 유저들의 rating 을 바탕으로 추천을 하는 방식이다.

Content-based 방식에서 u(c, s)를 구하기 위해 u(c, s<sub>i</sub>)를 이용했다면, Collaborative filtering 은 u(c, s)를 구하기 위해 u(c<sub>j</sub>, s)를 사용한다. 여기서 c<sub>j</sub>는 c와 "비슷한" 유저를 나타낸다. 즉, collaborative 방식에서는 유저의 취향이 유사하면, 그 아이템에 대한 평가도 유사할 것이라는 생각이 기반이 되는 것이다. 그래서 Collaborative 추천 방식에서는 유저 c와 비슷한 취향의 유저 `peer`를 찾는 것이 중요하다.

Collaborative 추천 방식은 크게 memory-based(=휴리스틱 베이스)방식과, model-based 방식으로 나뉘어진다.

## Memory-based

Memory-based 방식은 이전에 `rating`이 매겨진 모든 아이템에 대한 집합을 기반으로 휴리스틱하게 `rating`에 대한 예측을 하는 방식이다. 다음과 같이 Aggregate, Similarity 함수를 사용한다.

### Aggregate

유저 c에 대해서 아이템 s의 `rating`을 r<sub>c,s</sub>라고 하면 이는 다음과 같이 aggregate 를 통해 표현할 수 있다.

> r<sub>c,s</sub> = aggr(r<sub>c',s</sub>)  (c' : c와 가장 비슷한 유저 N개의 원소)

여기서 aggr(=aggregate) 함수는 그냥 집합을 합쳐서 무언가를 나타낸다는 의미로 이해하면 된다. 예를 들어 다음과 같은 식들을 적용할 수 있다.

1. 단순 평균
   
   > sum(r<sub>c',s</sub>) / N
2. 가중치 평균
   
   > sum(sim(c, c') * r<sub>c',s</sub>) / sum( | sim(c, c') | )  (sim : 비슷한 정도)
3. 편차에 대한 가중치 평균
   
   > r<sub>c</sub> + sum(sim(c, c') * (r<sub>c',s</sub> - r<sub>c'</sub>)) / sum( | sim(c, c') | )

여기서 sim(x, y) 함수는 유저 x와 유저 y가 유사한 정도로, 위에서는 유사한 유저의 `rating`을 높은 비중으로 반영하기 위한 용도로 쓰였다고 이해하면 된다. 3번 식은 유저마다 `rating`을 극단적으로 매기는 유저, 중간정도에서 매기는 유저, 낮은 점수 근처에서 매기는 유저 등 다양하게 있기 때문에 그런 유저의 특성을 보완하기 위해 편차로만 평균을 냈다고 생각하면 된다.

### Similarity

효과적인 aggregate를 위해 두 유저 사이의 유사정도를 구하기 위한 similarity 함수가 필요하다. 기본적으로 이 함수는 두 유저가 공통적으로 `rating`을 매긴 아이템들을 통해서 측정할 수 있다.

두 유저 x, y 사이의 공통 아이템들의 집합을 S<sub>xy</sub> 원소를 s라고 하면, 이 함수로 쓰일 수 있는 대표적인 예는 다음과 같다.

1. correlation
   > X = r<sub>x,s</sub> - r<sub>x</sub> , Y = r<sub>y,s</sub> - r<sub>y</sub> 라고 할 때,
   > sim(x, y) = sum(XY) / sqrt(sum(X^2)sum(Y^2))
   
2. 코사인 유사도 함수
   > sim(x, y) = sum(r<sub>x,s</sub>*r<sub>y,s</sub>) / sqrt(sum(r<sub>x,s</sub>^2)sum(r<sub>y,s</sub>^2))
   

결국 1번과 2번식의 차이는 평균과의 편차를 기준으로 하냐, 아니면 절대적인 `rating`을 기준으로 하냐의 차이가 된다. 1번 식은 평균보다 많이 좋거나 많이 나쁜 항목들이 유사할 때 유사도가 높고, 2번 식은 좋은 `rating`의 아이템을 비슷하게 평가할수록 유사하다는 것을 나타낸다.

## Model-based

Memory-based 방식은 유저의 `rating`데이터를 예측 식의 매개변수로 사용했었다. 그와 달리 model-based 방식은 유저의 `rating` 데이터를 모델 자체를 학습시키는데 사용한다.

간단한 예로 다음과 같이 확률분포를 통한 접근 방식이 있다.

### Probabilistic approach (확률론적 접근 방식)

Probabilistic approach 는 해당 모델이 확률모델로부터 추정값을 구하는 모델이라고 할 수 있다. r<sub>c,s</sub>에 대한 추정값을 E(r<sub>c,s</sub>)라 하면, E(r<sub>c,s</sub>)에 대해서 다음과 같이 식을 세울 수 있다.

> E(r<sub>c,s</sub>) = sum(i * Pr(r<sub>c,s</sub> = i | r<sub>c,s'</sub>)) (s' : 다른 아이템들)

이 식은 결국 r<sub>c,s</sub>가 i가 될 확률분포 Pr을 구하고, 이를 통해 r<sub>c,s</sub>를 예측하는 식이다. r<sub>c,s'</sub>는 이미 매겨진 다른 아이템이고, 이렇게 다른 아이템의 `rating` 상황이 주어졌을 때 (r<sub>c,s</sub> = i)에 대한 조건부확률을 구하는 것이다.

결국 여기서는 이 조건부 확률분포를 구하는 확률 모델이 중요한데, 대표적으로 **cluster model** 과 **Bayesian network** 가 있다.

1. **Cluster model**

   Cluster model 은 비슷한 성향의 유저를 묶어서 여러 클래스로 구분한 모델이다. 유저의 class membership 이 주어졌을 때, 유저 `rating`을 독립적으로 추정한다. 즉, 모델 구조는 naive Bayesian model 이 된다. 그리고 모델의 클래스의 개수와 변수값들 등은 데이터로부터 학습이 된다.

2. **Bayesian network**

   Bayesian network 는 각 `item`이 하나의 `node`가 된다. 그리고 각 `node`의 `state`가 바로 그 `node(=item)`에서 가능한 `rating`값들이 되어 각 조건부 확률을 구하게 된다.

이 모델들은 전부 데이터를 통해 조건부 확률을 학습하고, 그로부터 `rating`을 예측한다. 하지만 이러한 모델의 한계가 있는데, 첫번째로는 학습이 진행되면서 비슷한 유저끼리 묶이지 않는 경우가 발생할 수 있는데, 이는 유저끼리 모든 관심사가 비슷한게 아니라, 일부의 관심사만 공통될 수 있기 때문이다. 예를 들어 한 유저가 업무용 아이템과, 취미용 아이템을 쇼핑한 데이터가 있다면, 그 유저와 취미가 비슷하지만 다른 업무를 가진 유저는 다르게 클러스터될 수 있다.

## Collaborative 추천 방식의 한계

Collaborative 추천 방식으로 Content-based 추천방식의 문제점들을 일부 해결할 수 있다. 하지만 Collbaborative 방식만의 한계점 또한 존재한다.

1. **새로운 유저 문제**

   새로운 유저에 대해서는 정보가 없기 때문에 추천을 잘 해주기 어려운 것이 당연하다. 이는 content-based 방식과 동일한 문제점이다. 하지만, 결국 좋은 추천을 위해서는 이런 것들도 최대한 해결해야 한다. 이 문제를 보완하기 위한 접근방법으로 hybrid 방식 등을 사용한다.

2. **신규 아이템 문제**

   Collaborative 방식은 비슷한 취향의 유저가 선택했던 아이템들을 추천해주는 방식이다. 하지만 신규 아이템은 아직 아무 유저도 써보지 않았기 때문에 추천될 수가 없고, 결국 추천되지 않기때문에 신규 아이템은 계속 아무도 사용안하는 딜레마가 발생하게 된다. 이 문제도 hybrid 방식으로 보완될 수 있다.

3. **Sparsity(데이터 부족) 문제**

   데이터에서 sparse 란 값이 매우 띄엄띄엄 들어있는 경우를 말한다. 어떤 추천 시스템이든 보통 알고있는 `rating` 정보는 전체 아이템의 개수에 비해 매우 적다. 그래서 아주 적은 정보로 좋은 예측을 하기 위한 테크닉들이 필요한 것이다. 하지만 특히 collaborative 방식에서는 이 문제가 두드러지는데, collaborative 방식은 유저의 critical mass[^3]에 의존하기 때문이다.

   예를 들어 어떤 아이템을 경험한 유저수가 매우 적다면, 그 유저들이 모두 아이템을 좋게 평가했어도 해당 아이템은 추천받기가 어렵다. 왜냐하면 그 유저들과 비슷한 취향을 가진 유저들은 해당 아이템을 써보지 않았기 때문이다. 결국 아이템이 추천받기 위해서는 `rating`에 상관없이 최소한 일정 수준 이상 규모의 유저들이 써야 하는 것이다.

   또다른 예시로는, 독특한 취향의 유저가 있을 수 있다. 그 취향을 가진 유저 데이터가 드물다면, 그 유저는 거의 제대로운 추천을 받기가 어렵다.

   

   이 sparsity 문제를 해결하기 위해 나온 방법들로 다음과 같은 것들이 있다.
   
   - 유저간의 유사도를 계산할 때, 유저의 프로필 등 제3의 데이터를 활용하는 방법
   - 유저의 아이템과 관련된 최근 행동 기록등을 통해 유사도를 구하는 방법
   - 결손값을 구하기 위해 행렬을 이용하는 방법 (대표적으로 matrix factorization)

[^3]: 임계 질량. 어떤 일이 발생하기 위한 최소한의 크기. 여기서는 추천시스템이 제대로 동작하기 위한 유저수 정도의 뜻으로 이해하면 된다.