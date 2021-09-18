---
title: "ViT: An Image is Worth 16x16 Words - Transformers for Image Recognition at Scale"
tags: [논문, Vision, Transformer]
---

# 소개

이 논문은 비젼분야의 논문인데, BERT와 관련 자료를 찾던 도중, 재미있는 아이디어의 논문이어서 가져오게 되었다. 이 논문에서 제시하는 모델은 Vision Transformer (ViT) 라고 부르는데, 이름에서도 알 수 있듯이 [transformer](/attention-is-all-you-need)의 모델을 비젼분야에 도입한 것이다. 비젼 분야에 transformer나 attention 관련 아이디어를 도입한 논문은 많았지만, 결국 CNN기반의 성능을 쉽게 넘을 수 없었는데 ViT는 많은 SOTA를 갱신한 모델이다. 또한 아이디어 자체가 매우 간단하면서도 transformer의 장점을 효과적으로 이용했다고 생각한다. 내용 자체는 매우 간단하기 때문에 모델에 대해 간략히만 설명하도록 한다.

# 모델

먼저 transformer의 인코더를 떠올려보면, 이는 결국 여러 입력 토큰들을 임베딩시키고, 이 토큰간의 관계(attention)를 transformer 구조를 통해서 파악해서, 관계를 파악한 최종적인 임베딩 값이 나오게 된다. 그리고 그 임베딩값을 이용해서 태스크를 수행하는 것이다.

이 논문의 제목에서도 알 수 있듯이, ViT는 $16\times16$의 이미지를 하나의 토큰으로 본다. 결국, 그림을 퍼즐처럼 쪼개서, 이를 일렬로 나열된 하나의 문장으로 보고, 그 문장을 transformer를 통해서 해석한다. 여기에서는 이 이미지 토큰을 하나의 patch라고 표현하고 있다.

이 그림을 그냥 단순하게 일렬로 나열해서 문장을 만든다는 방식은 정말 간단한 아이디어지만, transformer를 정말 효과적으로 사용한 방법이었다고 생각한다. 먼저 언어 데이터와 이미지 데이터의 차이점은 다음과 같다.

이미지는 매우 많은 픽셀 수($W\times H$)와 적은 채널(RGB 3개)가 2차원 구조로 이루어져있다. 반면, 문장 데이터는 토큰의 수 자체는 이미지에 비해서 매우 적지만, 각 토큰마다 매우 많은 채널(임베딩 크기)로 표현한 후, 데이터를 학습시킨다.

ViT모델은 결국 patch를 통해서 이미지를 언어 문장처럼 임베딩 크기가 늘어나고, 개수가 줄어든 토큰으로 바꾸었다고 할 수 있다.

### Positional Embedding

논문에서는 Positional Embedding에 대해서 4가지 실험을 진행했다.

- Positional Embedding 자체를 주지 않음
- 기존 transformer와 마찬가지의 1차원 Positional Embedding
- 2차원 Positional Embedding
- Relative Positional Embedding (Offset에 따른 상대값)

결과적으로, Positional Embedding을 준 경우를 제외하고 나머지 Embedding은 퍼포먼스에 큰 영향이 없었고, 오히려 1차원의 경우가 미세하게 결과가 좋았다. 그래서 논문에서는 1차원 임베딩을 그대로 사용하였다.

이 부분을 생각해보면, 애초에 positional embedding 값 자체가 sinusoid를 통해 거리에 따른 상대적인 위치값이 표현되는 형태이기 때문에 큰 영향이 없는것도 이해할 수 있다. 입력이 이미지라는 점을 생각해보면, 조금만 회전시켜도 아래쪽에 위치하던 것이 오른쪽으로 옮겨갈 수도 있다. 그러기에 자세한 위치정보를 줘도 굳이 도움이 되지 않는다고 해석할 수 있겠다. 그래도 위치정보자체는 어느정도 주어야 한다는 점을 실험결과로 파악할 수 있었다.

### Patch Embedding

각 patch는 결국 $16\times16\times3$의 값을 갖게 되는데, 이를 linear projection을 통해 transformer 인코더의 latent vector 크기 $D$를 맞추어 주었다. 또한, 마지막에 classification을 위해서 BERT의 `[CLS]`토큰 아이디어를 활용해서 패치 임베딩뿐만 아니라 CLS Embedding도 넣어주었다. 이 CLS 임베딩에도 BERT와 마찬가지로 Positional Embedding 값이 들어가게 된다.

### Inductive Bias

Inductive bias란, 간단히 얘기하면 모델에서 입력 데이터의 특성을 미리 가정해서 제약을 걸어놓는 것을 의미한다. 이 논문에서는 CNN에 비해서 image-specific한 inductive bias를 훨씬 '덜' 사용했다고 얘기하고 있다. CNN은 기본적으로 locality, 즉 이미지에서 가까이 있을수록 관계가 있다는 전제가 기반이 되는데, ViT에서는 patch 이후로는 self-attention이 글로벌하게 적용되고 있기 때문에 CNN과 같은 bias가 많이 활용되지 않고 있다고 할 수 있다. 처음 패치를 구분지을 때만 local한 레이어가 사용된다.

### Hybrid Architecture

이미지에 대한 각 patch의 임베딩을 구할 때, raw image로부터 바로 선형변환을 했다. Hybrid Architecture는 선형변환 대신에 CNN을 활용하는 것을 말한다. CNN을 통해서 $D$차원만큼의 채널을 펼친다. CNN에 Pooling이 들어가기 때문에 결국 feature map 크기 자체가 작아지게 된다. 따라서 $1\times1$의 패치를 사용한다. 논문에서는 ResNet50 기반으로 Hybrid 구조를 적용했다.

결과적으로 이 Hybrid 모델은 모델크기가 작을 때는 더 좋은 결과가 나왔지만, 모델의 크기가 커질수록 순수한 ViT모델이 더 좋은 결과를 얻는다. CNN 전혀 없이 선형변환과 transformer만으로 좋은 결과를 얻어낸 것이다.

# 결론

논문에서는 레이어나 모델 크기, 종류, 임베딩 등 여러가지 부분에 대해서 많은 경우의 수를 실험했다. 실험 결과 중 언급할만한 요소들은 위의 모델 설명에 같이 추가하였다. 이 외에 나머지 시도들에 대한 내용과 결과는 생략하도록 한다.

논문에서 가장 아쉬운 점은 구글에서 낸 BERT같은 논문처럼 JFT-300M이라는 구글의 internal 데이터셋을 사용했다는 것이다. 이로 인해서, 이 모델의 결과를 다른 논문들과 완전히 동등하게 비교하기는 어렵다.

또한, patch에 대한 많은 실험들이 없었던 것들도 아쉽다. 결국에 이미지를 patch로 쪼개는 것이 핵심인데, 이 patch로 쪼개는 크기에 따라서도 임베딩을 만들 때 선형변환이 아닌 다른방식들을 사용할 수도 있다. 논문에는 모델의 다른 부분에 대한 실험은 많았지만, 비교적 patch크기나 선형변환방식을 사용하게된 이유가 잘 드러나지 않는 것이 아쉽다.

하지만 전체적으로 상당히 의미있는 논문이라고 생각이 된다. Vision분야에서는 CNN이 오랫동안 SOTA모델로 자리잡고, 갈수록 모델 자체에 대한 연구보다, EfficientNet등과 같이 미세한 성능을 향상시키기 위한 hyperparameter tuning등의 연구가 더 많이 이루어지고 있었다. Transformer 모델을 적용했던 기존의 많은 모델들도 결국 CNN과 결합시키는 등의 방법을 사용했었고, 기존 모델들의 결과를 쉽게 이기지는 못했다.

하지만 ViT에서는 아예 CNN을 전혀 사용하지 않고, 간단한 patch라는 개념을 통해서 transformer를 활용했고, 결국 좋은 결과를 얻었다. 이를 통해서 transformer에 대한 더 다양한 활용이 가능하다는 것을 제시해 준 것 같아서 비록 vision분야이지만 이 논문을 소개하게 되었다.
