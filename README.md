# Pokemon Classification

Transfer Learning을 이용하여 포켓몬 이미지를 분류하는게 목적인 프로젝트입니다.

## 데이터셋(from [Kaggle](https://www.kaggle.com/datasets/lantian773030/pokemonclassification))

- 데이터셋: PokemonData
- 클래스 수: 150개
- 총 이미지 수: 약 7000장



## 학습 설정

- 데이터 분할: Train 80% / Validation 20%
- 입력 크기: 224 × 224
- Batch Size: 32  
- Epoch: 10  
- Optimizer: Adam  
- Learning Rate: 0.001  
- Loss Function: CrossEntropyLoss  
- Evaluation Metric: Precision, Recall  

데이터 전처리:
- Train: RandomResizedCrop, RandomHorizontalFlip
- Validation: Resize, CenterCrop

---

## 실험 구성

총 4가지 설정으로 실험을 진행하였다.

| 실험 | 모델 | Pretrained | Feature Extract |
|---|---|---|---|
| 1 | ResNet34 | True | True |
| 2 | ResNet34 | False | True |
| 3 | ResNet34 | False | False |
| 4 | VGG16 | True | True |

---

## 실험 결과

| 모델 | 설정 | Precision | Recall |
|---|---|---:|---:|
| ResNet34 | pretrained + feature | 0.8401 | 0.8160 |
| VGG16 | pretrained + feature | 0.6640 | 0.6237 |
| ResNet34 | pretrained X + feature X | 0.3771 | 0.3592 |
| ResNet34 | pretrained X + feature | 0.0141 | 0.0220 |

- 최고 성능 모델: **ResNet34 (pretrained + feature extraction)**
- 최악 성능 모델: ResNet34 (pretrained X + feature extraction X)
---

## Learning Curve
### ResNet34(pretrained + feature)
![ResNet34(pretrained + feature)](test_result/resnet34_True_True_curve.png)

### VGG16(pretrained + feature)
![VGG16(pretrained + feature)](test_result/vgg16_True_True_curve.png)

### ResNet34(pretrained X + feature X)
![ResNet34(pretrained X + feature X)](test_result/resnet34_False_False_curve.png)

### ResNet34(pretrained X + feature)
![ResNet34(pretrained X + feature)](test_result/resnet34_True_False_curve.png)




---

## 결과 분석

- ResNet34가 VGG16보다 더 좋은 성능을 보였다.
- pretrained를 사용한 모델이 더 높은 성능을 보였다.<br>
  (데이터셋이 크지 않기에 미리 학습된 모델을 가져와 로컬데이터로 보정하는 쪽이 성능이 좋게 나온것같다.)
- pretrained없이 feature extraction을 사용하는 경우 매우 낮은성능을 보였다.<br>
  (Feature extraction은 pretrained 모델을 전제로 하기 때문에, 랜덤 초기화된 모델에서는 효과적으로 동작하지 않았다.)
  

## Streamlit 데모

학습된 모델을 기반으로 간단한 웹 데모를 구현하였다.

## 실행 방법<br>
**streamlit run app.py**
<br> or<br>
**python -m streamlit run app.py**


## 예측 결과 예시
![](demo_images/Bulbasaur_.PNG)
![](demo_images/Charmander_.PNG)
![](demo_images/Eevee_.PNG)
![](demo_images/squirtle_.PNG)

마우스로 그린 그림임에도 top5안에 정답이 있는 경우가 있는것으로보아 잘 작동하는것으로 보인다.<br>
인터넷에서 구한 사진으로 테스트시에 확실히 높은 정확도를 보여준다.


## 결론

- Transfer Learning은 작은 데이터셋에서 매우 효과적이다.
- 현재 데이터셋에서는 pretrained + feature extraction 조합이 가장 좋은 성능을 보였다.

