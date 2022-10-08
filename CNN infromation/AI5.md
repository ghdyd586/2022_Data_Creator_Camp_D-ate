[TOC]



# 합성곱 신경망의 구성 요소

## <SPAN STYLE = "COLOR:RED">합성곱 신경망(Convolution Neural Network)</SPAN>

> 필터를 사용하여 유용망 특성만을 드러나게 하여 이미지를 '압축'하는 방식을 사용

+ 완전 연결 신경망과의 차이

  > 챕터7에서 했던 완전 연결 신경망은 모든 입력에 각 가중치를 곱하고 절편을 더한다.
  >
  > 따라서 입력이 10개라면 10개의 출력이 생기고, 1000개라면 1000개의 출력이 생긴다.
  >
  > 일단 2차원 이미지 데이터를 옆에 픽셀과의 관계 상관 없이 1차원으로 잘라버리기 때문에 연관성이 끊어
  >
  > 질 수 있고 사용하는 노드(?), 입력(?)이 커진다. 컴퓨터가 많이 힘들어하고 정확도도 떨어지게 된다.

+ **`합성곱`**

<img src="https://user-images.githubusercontent.com/101400894/188311379-8aac818f-2dad-4db9-9fe3-59323abca469.png" alt="image" style="zoom:80%;" align="left"/>

+ 이전에는 모든 입력에 가중치를 곱했다면 합성곱은 필터의 크기만큼만 가중치를 곱해서 1개의 출력을 뽑아냄
+ 해당 필터는 1칸씩 이동하면서 나머지 출력도 만들어냄
+ 위는 1차원 합성곱



+ **`필터(=커널)`** : 입력의 위를 이동하면서 해당 데이터의 출력값을 만들어냄.
  + 합성곱에서는 뉴런이라고 말하지 않음
    + 입력층에 있는 모든 특성을 곱해서 구해지는 것이 뉴런

<img src="https://user-images.githubusercontent.com/101400894/188311853-0b6bbe3f-0ed1-441d-ad99-41530418f0c3.png" alt="image" style="zoom:80%;" align="left"/>

+ 2차원 필터에서는 입력 데이터가 4x4 짜리라면 3x3짜리 필터가 한 칸씩 이동하면서 출력값을 만들어낸다.
  + 합성곱 -> 1번째 자리* 1번째 자리.... 
  + 연산이 왼쪽에서 오른쪽, 위에서 아래로 연산이 진행됨
  + 합성곱연산시 마지막에 꼭 절편이 붙음

<img src="https://user-images.githubusercontent.com/101400894/188311623-5eaeae4c-3454-4911-818a-5402656fc43a.png" alt="image" style="zoom:80%;" align="left"/>

<img src="https://user-images.githubusercontent.com/101400894/188311578-3b939976-b397-42ca-afc4-a4194a4ad9bf.png" alt="image" style="zoom:67%;" align="left"/>

<img src="https://user-images.githubusercontent.com/101400894/188311754-ece60788-7f6e-4217-9cd8-ad2099ac4685.png" alt="image" style="zoom:80%;" align="left"/>

+ 필터를 통해 입력값이 출력될 때 그 출력을 특성 맵이라 부름
+ 필터를 사용해 특성맵을 만들때 활성화 함수를 사용함(렐루)
+ 필터를 여러개 사용하면 다양한 특성맵이 나올 수 있음 -> (2, 2, 3)출력이 나올 수 있음(필터의 개수가 3차원)



### 케라스 합성곱 층

>  `keras.layers` 에 모두 구현되어 있다. 먼저 2D 입력은 Conv2D 클래스를 사용`tf.keras.layers.Conv2D(필터개수, 필터사이즈, 활성화함수)` 로 사용

```python
from tensorflow import keras
keras.layers.Conv2D(10, kernel_size = (3, 3), activation = 'relu', input_shape(28,28,1))
```

+ input_shape은 처음에 합성곱 층을 만들때는 입력해야함
  + 나중에 따로 build 메서드를 따로 호출하거나 input_shape를 입력해야함
  + 배치차원은 x(이미지의 개수는 x)



#### <span style = "color:blue">패딩(padding)</span>

>  합성곱 연산 전에 입력 데이터의 주변을 특정 값으로 채워서 결과값의 크기, 픽셀별 가중치를 조정할 수 있다.

- 주변에 1개의 픽셀을 추가하여 슬라이딩을 더 많이하도록 해서 특성맵의 크기를 늘림
- 주로 입력과 출력의 특성맵의 크기가 같도록 만듦
- 패딩되는 주변의 값은 0

<img src="https://user-images.githubusercontent.com/101400894/188312191-1e2ef85d-6cdb-4f57-9bf2-353078fb544e.png" alt="image" style="zoom:80%;" align="left"/>

+ **패딩의 목적**
  + 패딩을 안하게 되면 아래처럼 가장 가장자리의 입력값이 중간에 있는 값과 비교해서 덜 사용되기 때문
  + 가장자리에 있는 입력 데이터가 합성곱에 참여하는 비율을 늘릴 수 있음
  + 좌측은 중앙과 4배 차이가 나지만 패딩을 한 우측은 2.25배 차이로 줄었음

<img src="https://user-images.githubusercontent.com/101400894/188312288-17c3dfd8-9874-4c0b-8803-b11daf19e5d4.png" alt="image" style="zoom:80%;" align="left"/>

```python
from tensorflow import keras
keras.layers.Conv2D(10, kernel_size = (3, 3), activation = 'relu', padding = 'same')
#입력과 출력의 사이즈가 같아지도록 패딩의 크기를 추가해줌
#'valid' -> 패딩을 전혀 사용하지 않은 경우
```



#### <span style = "color:blue">스트라이드(Strides)</span>

> 오른쪽, 아래쪽으로 이동하는 크기가 기존에는 1칸씩이지만 이것으로 수정할 수 있음
>
> 1보다 크게 사용하는 경우가 드물어서 default 1을 수정하지 않는 경우가 대부분이다.

<img src="https://user-images.githubusercontent.com/101400894/188312474-941866c1-cfb9-4152-9608-222db4fad9e0.png" alt="image" style="zoom:80%;" align="left"/>

+ 위 그림은 왼쪽->오른쪽, 위 -> 아래의 이동이 2칸씩 이동
+ 패딩을 해도 stride를 2로 지정하면 특성맵의 크기가 줄어듬

```python
from tensorflow import keras
keras.layers.Conv2D(10, kernel_size = (3, 3), activation = 'relu',
					padding = 'same', strides = 1)
#스트라이드의 크기 = 1
```

<img src="https://user-images.githubusercontent.com/101400894/188312601-dbaa1522-4617-40e9-b63f-851b21fe7b41.png" alt="image" style="zoom:50%;" align="left"/>



#### <span style = "color:blue">풀링(Pooling)</span>

> 특성맵의 세로/가로 크기를 줄이는 연산 -> 데이터를 압축함. 
>
> 최대풀링, 평균풀링이 있으며 주로 최대 풀링(MaxPooling)을 사용한다.

+ 보통 패딩을 통해 같은 크기의 출력을 만들어주고 풀링해서 특성맵의 크기를 줄임
+ 아래의 그림에서 3개의 필터 사용 -> 특성맵이 (2, 2, 3) -> (2, 2) 풀링을 적용해 1개의 특성뱁 3개가 생성
+ 평균풀링이면 입력 특성맵의 평균을 출력, 최대 풀링은 최댓값을 출력**(가중치가 없음)**
+ 필터 여러개에 의해 만들어진 특성맵의 개수 자체를 줄일 순 없음
+ 풀링은 겹치지 않게 필터 적용

![image](https://user-images.githubusercontent.com/101400894/188312709-c106b926-01e8-4fd5-989b-8768689831f8.png)

```python
from tensorflow import keras
#풀링의 크기
keras.layers.MaxPooling2D(2)
#strides는 구지 지정하지 않아도 됨(자동으로 풀링의 크기에 맞춰 스트라이드가 세팅됨)
#패딩은 크기를 줄이는게 목적이라 패딩을 valid 설정
keras.layers.MaxPooling2D(2, strides=2, padding='valid')
```



### 합성곱 신경망 전체 구조

![image](https://user-images.githubusercontent.com/101400894/188313469-c9dbaded-b602-48e9-813a-aa98b95b6667.png)

> 입력 데이터에 패딩을 설정하고 n개의 필터로 특성 맵 n개를 만든다. 
>
> 특성맵을 최대 풀링을 사용하여 데이터를 압축하고 이후에 Flatten()으로 쭈욱 나열시켜서 밀집층으로 전달한다.

+ 출력층
  + 분류라면 분류 개수에 맞춰서 출력층 개수 조절



### 3차원 이미지를 위한 합성곱

> RGB 채널로 구성된 이미지는 3차원이다. 
>
> 이것은 3차원 데이터를 필터를 할 때 3장을 꾸욱 찍어낸다고 생각하면 좋다. 그 예시는 다음과 같다.

![image](https://user-images.githubusercontent.com/101400894/188313603-bb5aaf0f-10c5-455f-9a02-a1c1e2e64d48.png)

+ 흑백 이미지도 채널이 1개 있다고 봐도 됨

![image](https://user-images.githubusercontent.com/101400894/188313743-88d9c301-b49f-448b-8762-21be8a1e8d9b.png)



## CNN을 이용한 이미지 분류

### 이미지 분류

> fashion 데이터를 가져와서 앞에 배운 합성곱 신경망



#### 기본 데이터 준비

```python
from tensorflow import keras
from sklearn.model_selection import train_test_split

#케라스에서 데이터를 받아옴(이때 훈련세트와 테스트세트를 나눠서 가져옴)
(train_input, train_target), (test_input, test_target) = \
	keras.datasets.fation_mnist.load_data()

#keras api의 합성곱 클래스는 입력데이터가 채널차원이 있을거라 알고 있음
#이미지를 (28, 28, 1)로 reshape해서 넘김(이때 차원이 1 늘었다고 해서 집약되거나 하는게 없음)
#이미지 데이터는 픽셀이라 255를 나눠서 정규화
train_scaled = train_input.reshape(-1, 28, 28, 1)/255.0

#훈련, 검증세트 나누기
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)
```

+ 데이터 reshape
+ 패션 데이터가 흑백인데 흑백은 위에서 설명했듯이 채널이 1개 있는 모양

<img src="https://user-images.githubusercontent.com/101400894/188313889-86be28ba-0adf-4848-b2ee-8f0c22a7ab80.png" alt="image" style="zoom:80%;" align="left"/>



#### <span style = "color:red">합성곱 신경망 만들기</span>

+ 기본 진행 그림으로 확인하기

![image](https://user-images.githubusercontent.com/101400894/188314089-424b2810-7577-4670-95e0-e2917576dce3.png)

1. 케라스 Sequential() 메서드를 사용해서 층 추가
2. 첫 층은 Conv2D층 사용(필터) -> 필터개수가 32개라서 특성맵도 depth가 32
3. 그 다음 MaxPooing2D를 사용해서 풀링(이것도 층)

![image](https://user-images.githubusercontent.com/101400894/188314798-b58badec-cec1-4f30-9dbd-a18e67b0207b.png)

```python
#층 만들 준비
model = keras.Sequential()

#합성곱층
#kernel_size는 채널에 대한 입력이 없이 자동으로 입력값의 채널의 크기와 동일하게 구성됨
#여기서는 흑백이미지라 채널이 1이라서 kerner_size =3 이므로 (3, 3, 1)이 될 것임
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', 
                              padding='same', input_shape=(28,28,1)))

#풀링층
model.add(keras.layers.MaxPooling2D(2))

#합성곱층
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', 
                              padding='same'))
#풀링층
model.add(keras.layers.MaxPooling2D(2))

#풀링한 것을 1차원 배열로 나열(현재 3,136개의 값이 있음)
model.add(keras.layers.Flatten())

#Dense층(은닉층 -> 100개)
model.add(keras.layers.Dense(100, activation='relu'))

#과대적합을 막기위해 Dropout, 하이퍼파라미터는 40%정도
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 3136)              0         
                                                                 
 dense (Dense)               (None, 100)               313700    
                                                                 
 dropout (Dropout)           (None, 100)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 333,526
Trainable params: 333,526
Non-trainable params: 0
_________________________________________________________________
'''

keras.utils.plot_model(model)
#크기의 변형까지 볼려면 show_shapes=True
keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)
```



#### 컴파일과 훈련

```python
#완전연결신경망이나 합성곱신경망이나 컴파일과 훈련 방법은 동일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

model.evaluate(val_scaled, val_target)
#375/375 [==============================] - 3s 8ms/step - loss: 0.2184 - accuracy: 0.9205
#[0.21837739646434784, 0.9204999804496765] [손실값, 정확도]

#이미지 하나 보기
#depth 차원 없애고 흑백반전으로 보기
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

#첫번째 이미지의 예측 확률 보기
#10개의 확률이 나옴
#배치차원을 유지한 채로 그림을 뽑아와야되기 때문에
#단순히 val_scaled[0]이 아닌 val_scaled[0:1]을 사용(1*28*28*1의 이미지가 전달)
preds = model.predict(val_scaled[0:1])
print(preds)
#1/1 [==============================] - 0s 138ms/step
#[[6.7846774e-12 8.1426743e-22 8.9696543e-16 7.7117090e-15 6.6757140e-14
#  1.4335832e-13 3.7601382e-14 3.6749163e-12 1.0000000e+00 1.8052020e-13]]

#막대그래프 그리기
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

import numpy as np
print(classes[np.argmax(preds)])
#가방

#테스트 세트도 똑같이 정규화하기
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

#점수확인
model.evaluate(test_scaled, test_target)
#313/313 [==============================] - 3s 9ms/step - loss: 0.2423 - accuracy: 0.9156
#[0.24227263033390045, 0.9156000018119812]
```

<img src="https://user-images.githubusercontent.com/101400894/188315170-ac3f7387-b839-4483-9ef2-0a7ad792f44c.png" alt="image" style="zoom:67%;" align="left"/><img src="https://user-images.githubusercontent.com/101400894/188315261-784f151c-9d35-414b-bc66-20c67a41c67f.png" alt="image" style="zoom:67%;" />



## 합성곱 신경망의 시각화

### <span style = "color:red">가중치의 시각화</span>

![image](https://user-images.githubusercontent.com/101400894/188315495-dbf5d7b4-b28e-4023-aa05-3c1437f9ea52.png)

+ 입력 * 필터 -> 특성맵 만들기
+ 필터가 어떤 특징을 학습했다 한다면 가중치가 높은 영역과 낮은 영역으로 필터가 채워짐
  + 이미지가 스캔했을때 비슷한 영역에 가중치가 높게 나오면 높은 출력값이 나옴



#### 이전에 저장한 데이터 불러오기

```python
from tensorflow import keras

# 코랩에서 실행하는 경우에는 다음 명령을 실행하여 best-cnn-model.h5 파일을 다운로드받아 사용하세요.
#이전 모델 받아오기
!wget https://github.com/rickiepark/hg-mldl/raw/master/best-cnn-model.h5
'''
--2022-05-19 01:28:27--  https://github.com/rickiepark/hg-mldl/raw/master/best-cnn-model.h5
Resolving github.com (github.com)... 140.82.113.4
Connecting to github.com (github.com)|140.82.113.4|:443... connected.
HTTP request sent, awaiting response... 302 Found
Location: https://raw.githubusercontent.com/rickiepark/hg-mldl/master/best-cnn-model.h5 [following]
--2022-05-19 01:28:28--  https://raw.githubusercontent.com/rickiepark/hg-mldl/master/best-cnn-model.h5
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.111.133, 185.199.108.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4049416 (3.9M) [application/octet-stream]
Saving to: 'best-cnn-model.h5.2'

best-cnn-model.h5.2 100%[===================>]   3.86M  --.-KB/s    in 0.08s   

2022-05-19 01:28:28 (47.6 MB/s) - 'best-cnn-model.h5.2' saved [4049416/4049416]
''''

model = keras.models.load_model('best-cnn-model.h5')
```



#### model.layers

>  `model.layers` 으로 불러오면 층에 대한 정보들이 나옵니다. 
>
> 첫번째 합성곱층의 가중치를 봅시다. 
>
> weights 속성에서 인덱스 0은 가중치이고 인덱스 1은 절편입니다. 
>
> 모양으로 보아하니 (3, 3, 1, 32) 으로 잘 설정된 것 같습니다. 
>
> 절편의 개수도 필터마다 1개 절편이므로 (32,) 로 잘 나왔습니다.
>
> weights 속성으로 평균과 표준편차를 볼 수 있습니다.
>
> 가중치 분포를 히스토그램으로 확인해보면 다음과 같습니다. 
>
> 훈련하지 않은 합성곱 신경망과 비교를 해봅시다.

```python
model.layers
'''
[<keras.layers.convolutional.conv2d.Conv2D at 0x7f1da1df4250>,
 <keras.layers.pooling.max_pooling2d.MaxPooling2D at 0x7f1da1df4fa0>,
 <keras.layers.convolutional.conv2d.Conv2D at 0x7f1da0533730>,
 <keras.layers.pooling.max_pooling2d.MaxPooling2D at 0x7f1da04ec4f0>,
 <keras.layers.reshaping.flatten.Flatten at 0x7f1da04fe9a0>,
 <keras.layers.core.dense.Dense at 0x7f1da04fef40>,
 <keras.layers.regularization.dropout.Dropout at 0x7f1da04feca0>,
 <keras.layers.core.dense.Dense at 0x7f1da0506d30>]
'''

#특정 층 정보 불러오기
conv = model.layers[0]

#weights속성 -> 합성곱층 만들때 사용한 필터와 가중치가 포함
#[0]에는 가중치 정보, [1]에는 절편
print(conv.weights[0].shape, conv.weights[1].shape)
#(3, 3, 1, 32) (32,)  // 3,3,1 이미지(흑백이미지), 필터 32개, 절편도 32개

#가중치만 보기
conv_weights = conv.weights[0].numpy()

print(conv_weights.mean(), conv_weights.std())
#-0.021033935 0.23466988

import matplotlib.pyplot as plt

#reshape을 사용해서 하나의 열벡터가 되도록 펼치고 히스토그램 그려보기
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()
```

+ 훈련된 가중치 개수와 훈련되지 않은 가중치의 개수
  + 가중치가 0이 값들은 이미지에서 어떠한 의미있는 부분이 있었다는 뜻

<img src="https://user-images.githubusercontent.com/101400894/188316074-0ca59f87-6b4c-42db-84cd-775cffa5b838.png" alt="image" style="zoom:67%;" />



#### 층의 가중치 시각화

```python
fig, axs = plt.subplots(2, 16, figsize=(15,2))

for i in range(2):
    for j in range(16):
    	#(3, 3, 1, 32)
    	#출력되는 32개의 가중치가 동일한 기준을 가지도록 vmin과 vmax로 설정
        axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')

plt.show()
```

<img src="https://user-images.githubusercontent.com/101400894/188316324-9b306443-b8e1-4757-9a8a-aee329ae2527.png" alt="image" style="zoom:67%;" align="left"/>



#### 훈련되지 않은 모델 보기

```
no_training_model = keras.Sequential()

no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', 
                                          padding='same', input_shape=(28,28,1)))
no_training_conv = no_training_model.layers[0]

print(no_training_conv.weights[0].shape)
#(3, 3, 1, 32)
no_training_weights = no_training_conv.weights[0].numpy()

print(no_training_weights.mean(), no_training_weights.std())
#-0.0029798597 0.08092386

plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

fig, axs = plt.subplots(2, 16, figsize=(15,2))

for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')

plt.show()
```



### <span style = "color:red">함수형 API</span>

<img src="https://user-images.githubusercontent.com/101400894/188316592-2b2937e5-9f02-4781-a93d-c20de6f7d83a.png" alt="image" style="zoom:100%;" align="left"/>

+ 함수형 api는 dense1객체를 함수처럼 호출하는 것
+ 파이썬의 모든 객체들은 함수형 api처럼 호출할 수 있음
+ keras.Sequential()객체를 만들면 기본적으로 inputLayer층이 생김
+ dense1(inputs)는 inputLayer층에 해당하는 객체를 넣는 곳
+ keras.input(shape = ...)은 input메서드를 이용해서 InputLayer 클래스 객체를 반환
  + inputs 객체를 shape 메서드로 크기 지정을 해줌

```python
print(model.input)

'''
KerasTensor(type_spec=TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='conv2d_input'), name='conv2d_input', description="created by layer 'conv2d_input'")
'''

conv_acti = keras.Model(model.input, model.layers[0].output)
```



#### 특성맵 시각화

> 케라스로 중간의 Conv2D 층이 출력한 특성 맵을 가져와서 한 샘플을 입력해서 그 결과를 확인
>
>  same 패딩과 32개 필터를 사용했으므로 특성맵의 크기는 (28, 28, 32) 이다.

<img src="https://user-images.githubusercontent.com/101400894/188317097-a7abf4a1-3a67-4b63-9efa-71640ceb68e8.png" alt="image" style="zoom:100%;" align="left"/>

```python
#input은 inputLayer 객체
#model.layers[0].output은 합성곱층(Conv2D)
conv_acti = keras.Model(model.input, model.layers[0].output)

#데이터 가져오기
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

#데이터 하나 보기
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

#데이터 하나를 뽑아서 정규화
inputs = train_input[0:1].reshape(-1, 28, 28, 1)/255.0

#뽑아낸 데이터의 특성맵 뽑아내기
feature_maps = conv_acti.predict(inputs)
#1/1 [==============================] - 0s 235ms/step

print(feature_maps.shape)
#(1, 28, 28, 32)

#그림 보기
fig, axs = plt.subplots(4, 8, figsize=(15,8))

for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')

plt.show()
```

<img src="https://user-images.githubusercontent.com/101400894/188317331-46e6eb62-536d-41c5-9b7c-b1b647be6547.png" alt="image" style="zoom:50%;" align="left"/>



#### 두번째 합성곱층 확인하기

> 시각적으로 무엇을 했는지 이해가 가지 않지만 이미 한 층을 거렸으므로 시각적인 정보보다는 추상적인 정보를 학습한다고 봐야합니다. 층을 여러개 넣을 수록 비슷한 결과를 볼 수 있을 것이다.

```python
#input은 유지, model.layers[2] -> Conv2D
conv2_acti = keras.Model(model.input, model.layers[2].output)

feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1)/255.0)
#1/1 [==============================] - 0s 72ms/step

print(feature_maps.shape)
#(1, 14, 14, 64)


fig, axs = plt.subplots(8, 8, figsize=(12,12))

for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')

plt.show()
```

<img src="https://user-images.githubusercontent.com/101400894/188317394-cffd80e3-e1a6-499c-98f7-3dd7c8d9ad5e.png" alt="image" style="zoom:50%;" align="left"/>



# 텍스트를 위한 인공신경망

## 순차 데이터와 순환 신경망

### 순차 데이터(Sequential Data)

> 텍스트, 시계열 데이터(Time Series Data, 일정 시간 간격으로 기록된 데이터) 처럼 순서에 의미가 있는 데이터

+ 이미지 한 장을 분류하는 것은 순서가 상관 없고 오히려 모델을 만들 때 골고루 있는 것이 General 한 모델이 되므로 더 좋다. 
+ 그러나 댓글과 같은 텍스트 데이터는 글의 순서가 중요한 순차 데이터이고 순서를 유지해주어야 한다.(ex : '별로지만 추천해요' 를 잘라서 다른 것들과 섞이게 하면 '추천해요' 와 '별로지만' 이 된다. 붙여서 분석해야 '긍정'이 아니라는 것을 알 수 있습니다.)

- **`피드포워드 신경망(FeedForward Neural Network)`** 는 입력 데이터의 흐름이 앞쪽으로만 전달되는 신경망

챕터7의 완전신경망, 챕터8의 합성곱 신경망이 FFNN에 속한다. 신경망이 이전에 처리했던 샘플을 다음 샘플 처리를 위해 재사용하기 위해서는 다른 개념이 필요하다. 바로 '순환 신경망' 이다.



### <span style = "color:red">순환 신경망(Recurrent Neural Network, RNN)</span>

> 순환 신경망(Recurrent Neural Network, RNN) 은 완전 연결 신경망과 거의 비슷하지만 이전 데이터를 순환하도록 하는 고리를 붙인다.

<img src="https://user-images.githubusercontent.com/101400894/188317905-aad19544-06ce-4679-b77c-776349b4217b.png" alt="image" style="zoom:80%;" align="left"/>

+ 따라서 다음 입력을 처리하는 과정에서 이전의 데이터들을 어느정도 포함하고 있을 것입니다. '이전 샘플에 대한 기억을 가지고 있다' 라고 말한다고 합니다.

+ 3차원의 배열로 들어올것으로 기대

  - 타임스텝(time step) : 샘플을 처리하는 한 단계

  - 셀(cell) : 순환 신경망에서의 층. 한 셀에는 여러개 뉴런이 있으나 모두 표현X( == 순환층)

  - 은닉 상태(hidden state) : 셀의 출력
  - 활성화 함수 : tanh

> 셀 뒤에 붙은 회색 작은 동그라미는 활성화함수 입니다. 
>
> 일반적으로 활성화 함수는 하이퍼볼릭 탄젠트(hyperbolic tangent, tanh) 함수가 사용되며 s자 형태이기 때문에 종종 시그모이드 함수라고 부른다고 합니다. 
>
> 다만 시그모이드의 치역은 0~1, 하이퍼볼릭 탄젠트는 -1~1 이기 때문에 구분이 필요
>
> 순환 신경망의 활성화 함수는 입력, 이전 타임스텝의 은닉 상태 입력에 각각 가중치를 곱하여 결과를 도출해냅니다. (가중치가 여러개 라는 뜻)



#### 타임스텝으로 펼친 신경망

![image](https://user-images.githubusercontent.com/101400894/188318390-f7f0765c-1f87-4a2b-93a7-2bcb70d555ee.png)

+ 가중치의 공유
  + w_x와 입력 x를 곱해서 h를 만듬(w_x는 계속 사용됨)
  + h의 가중치 w_h와 h를 곱해서 새로운 은닉상태를 생성....



#### 순환 신경망의 가중치

> CNN은 합성곱으로 데이터를 압축한 인공신경망이라고 정리했다. 순환신경망은 상당히 다르다. 각 입력층이 순환층으로 완전연결되어있고 순환층의 결과가 각 다른 셀에 영향을 끼친다. 

<img src="https://user-images.githubusercontent.com/101400894/188318537-72b75cc1-5880-4e35-82b3-d91b8e7e4042.png" alt="image" style="zoom:80%;" align="left"/><img src="https://user-images.githubusercontent.com/101400894/188318586-ecaf5cbc-1999-42ba-af5e-0807069ac99d.png" alt="image" style="zoom:33%;" />

+ 4개의 입력이 있고 3개의 순환층이 있다면 12개의 가중치가 존재(절편 제외)
+ r_1, r_2, r_3는 각각 자신이 아닌 다른 뉴런으로도 완전 연결됨(오른쪽 그림)



#### 순환 신경망의 입력

<img src="https://user-images.githubusercontent.com/101400894/188318823-09776165-c731-40e5-a158-70e66078b01d.png" alt="image" style="zoom:100%;" align="left"/>

+ 하나의 **샘플** = 문장이라 가정
+ 4개의 **단어(토큰)**
+ 각각 하나의 단어를 여러개의 실수(정수 값)로 표현 가능(3개의 원소를 가진 벡터로 표현)
  + (1, 4, 3) -> 1문장(샘플)에 단어 4개를 3개의 실수및 정수로 표현



#### 다층 순환 신경망

<img src="https://user-images.githubusercontent.com/101400894/188318935-87187640-1bbd-496d-accc-106ae513eb11.png" alt="image" style="zoom:80%;" align="left"/>

> 어떤 순환 신경망이 순환했다면 마지막 타임스텝의 은닉상태만 출력됨
>
> 이전의 모든 순환 신경망의 타임스텝의 셀은 모든 타임스텝의 은닉상태를 출력해야 함



#### 순환 신경망을 사용한 예측

<img src="https://user-images.githubusercontent.com/101400894/188319160-17265cbe-db7b-4c05-b5e3-af8c1c42b308.png" alt="image" style="zoom:80%;" align="left"/>

> 만약 샘플이 10개, 타임스텝이 20개, 각 벡터가 100인 입력을 나타낸다고 하면 (10, 20, 100) 으로 나타낼 수 있고 순환층을 연결해서 사용한다고 하면
>
> > (10, 20, 100) --> 순환층 --> (10, 20, 순환층의 뉴런 개수) --> 순환층 --> 마지막 타임스텝의 은닉상태



## 순환 신경망으로 IMDB 리뷰 분석하기

> IMDB : 영화 평점 사이트



### IMDB 리뷰 데이터셋

>  영화 사이트 리뷰 데이터셋으로 각각 훈련, 테스트 각각 25,000개씩 총 50,000 개 데이터셋이 있다. 0과 1로 라벨링이 되어있는데 이는 `긍정, 부정을 뜻하며 각각 50%, 50%`이다.
>
> [IMDB(Keras)](https://keras.io/api/datasets/imdb/)

<img src="https://user-images.githubusercontent.com/101400894/188443261-9101656b-07a1-4a8e-845f-7edc70aa6c59.png" alt="image" style="zoom:67%;" align="left"/>

+ NLP : 텍스트 데이터
+ 말뭉치 : 데이터셋
+ 토큰 : 분리된 단어 하나. 하나의 타임스텝을 의미
  + 이 토큰들이 각각 정수에 매핑되어 있고, 0은 패딩, 1은 시작, 2는 어휘사전에 없는 토큰에 해당
  + 위의 keras.imdb 에 들어가보니, 사용 빈도로 인덱싱 되어있음
    + 우리는 상위 500개만 사용
  + 리뷰의 길이(토큰 개수)를 평균과 중간값으로 확인
+ 어휘사전 : 말뭉치에 나오는 고유한 단어, 토큰의 집합



#### 케라스로 IMDB 데이터 불러오기

```python
#데이터가 단어지만 단어가 이미 숫자로 다 표시가 되어있음
from tensorflow.keras.datasets import imdb

#로드하면서 훈련세트, 테스트세트 나누기, 어휘사전은 500개의 단어만 사용
(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)
    
print(train_input.shape, test_input.shape)
#(25000,) (25000,)

print(len(train_input[0]))
#218

print(len(train_input[1]))
#189

print(train_input[0])
'''
# 1은 문장 시작, 2는 어휘사전에 없는 토큰(나중에 채워야 함)
# 파이썬 리스트인 것을 확인 가능
[1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 2, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 469, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2, 2, 16, 480, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 476, 26, 400, 317, 46, 7, 4, 2, 2, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]
'''

print(train_target[:20])
#[1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]
```

<img src="https://user-images.githubusercontent.com/101400894/188446195-ae9bf25d-1ab8-423e-bf7a-68a7c32217f3.png" alt="image" style="zoom:80%;" align="left"/>



#### 훈련세트 준비

```python
from sklearn.model_selection import train_test_split

#검증 세트 20% 떼어내기
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

import numpy as np

# train_input을 모두 순회하면서 길이를 파이썬 리스트로 만들고 그것을 넘파이배열로 만듬
lengths = np.array([len(x) for x in train_input])

print(np.mean(lengths), np.median(lengths))
#239.00925 178.0

import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
```



#### 시퀸스 패딩

<img src="https://user-images.githubusercontent.com/101400894/188447010-316f6a06-6673-4f1c-8b62-95a7421d575a.png" alt="image" style="zoom:80%;" align="left"/>

> 시퀀스 데이터 길이를 100으로 맞추rh maxlen 옵션을 지정하여 길면 자르고, 짧으면 0으로 패딩
>
> 이를 시퀸스 패딩이라 함

```
from tensorflow.keras.preprocessing.sequence import pad_sequences

#maxlen을 100으로 지정해서 100자를 넘으면 자르고 짧으면 0 추가
train_seq = pad_sequences(train_input, maxlen=100)

#25000개에서 5000개는 검증세트로 떼어냄
print(train_seq.shape)
#(20000, 100)

print(train_seq[0])
'''
[ 10   4  20   9   2 364 352   5  45   6   2   2  33 269   8   2 142   2
   5   2  17  73  17 204   5   2  19  55   2   2  92  66 104  14  20  93
  76   2 151  33   4  58  12 188   2 151  12 215  69 224 142  73 237   6
   2   7   2   2 188   2 103  14  31  10  10 451   7   2   5   2  80  91
   2  30   2  34  14  20 151  50  26 131  49   2  84  46  50  37  80  79
   6   2  46   7  14  20  10  10 470 158]
'''

#train_input[0]는 100자보다 김(끝이 0으로 마무리가 되지 않음)
#끝부분을 출력해봤는데 train_seq[0]의 끝부분과 같은것으로 보아 앞부분을 자른것을 확인
print(train_input[0][-10:])
#[6, 2, 46, 7, 14, 20, 10, 10, 470, 158]

#패딩도 뒷부분이 아닌 앞부분에 0을 채움
print(train_seq[5])
[  0   0   0   0   1   2 195  19  49   2   2 190   4   2 352   2 183  10
  10  13  82  79   4   2  36  71 269   8   2  25  19  49   7   4   2   2
   2   2   2  10  10  48  25  40   2  11   2   2  40   2   2   5   4   2
   2  95  14 238  56 129   2  10  10  21   2  94 364 352   2   2  11 190
  24 484   2   7  94 205 405  10  10  87   2  34  49   2   7   2   2   2
   2   2 290   2  46  48  64  18   4   2]

#검증세트도 위와 같이 설정
val_seq = pad_sequences(val_input, maxlen=100)
```



### 순환 신경망 모델 만들기

```
from tensorflow import keras

#Sequential()을 이용해서 모델을 만듬
model = keras.Sequential()

# 가장 기본적인 RNN 순환신경망은 keras.layers.SimpleRNN
# 맨 첫번째 RNN에 들어있는 뉴런의 개수 지정
# input_shape을 지정(입력값의 형태), 길이가 100
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))

# 은닉층(RNN층 다음에 은닉상태를 펼칠 필요가 없음(Flatten 필요x))
model.add(keras.layers.Dense(1, activation='sigmoid'))

train_oh = keras.utils.to_categorical(train_seq)

print(train_oh.shape)
#(20000, 100, 500)

print(train_oh[0][0][:12])
#[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]

print(np.sum(train_oh[0][0]))
#1.0

val_oh = keras.utils.to_categorical(val_seq)

model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 8)                 4072      
                                                                 
 dense (Dense)               (None, 1)                 9         
                                                                 
=================================================================
Total params: 4,081
Trainable params: 4,081
Non-trainable params: 0
_________________________________________________________________
'''
```

+ keras.layers.SimpleRNN()
  + 매개변수
    + 첫번째 : 뉴런의 개수
    + input_shape : 입력값의 형태 (타임스텝의 수, )



+ 원-핫 인코딩
  + train_seq에서 단어(토큰)은 1, 4, 5, ... 등의 숫자로 표시된 것을 알 수 있음
    + 이런 단어들이 서로 상호 관계없는 무의미한 방식으로 인코딩하는 것이 `원-핫 인코딩`
    + 현재 500개의 단어사전을 준비했기 때문에 원-핫 인코딩을 만들때 500개의 벡터로 이루어진
      원핫 인코딩을 만들어야 함
    + utils.to_categorical(train_seq) -> train_seq = (20000, 100)
      + 원핫 인코딩이 어휘사전이 500개이기 때문에 500개의 벡터가 나옴
      + (20000, 100, 500)

<img src="https://user-images.githubusercontent.com/101400894/188448893-e9d9be11-d5bc-46bd-8c2a-8941c9fbd994.png" alt="image" style="zoom:80%;" align="left"/>



+ 모델 구조
  + 모델의 파라미터 : 입력 토큰 500개 x 순환층 뉴런 8개 + (순환파트)은닉상태 크기 8개 x 뉴런 8개 + 절편 8개 = 4072개 
  + Dense층의 가중치 개수 9개까지 더해서 총 4081개



#### 모델 훈련

```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', 
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

<img src="https://user-images.githubusercontent.com/101400894/188452210-10f7528e-6d93-45e9-9959-d9501e21af27.png" alt="image" style="zoom:67%;" align="left"/>



### 단어 임베딩을 사용하기

> RNN 에서 텍스트 처리를 할 때에는 '단어 임베딩(Word Embbeding)' 을 사용
>
> 단어를 고정된 크기의 실수 벡터로 바꾸어주는 방법인데 단어의 의미를 고려하여 벡터로 표현한 것
>
> 예를 들어, [고양이, 강아지, 개구리] 가 있다면 각각을 벡터로 표현하지만 고양이와 강아지 사이의 거리가 
>
> 개구리와의 거리보다 가깝습니다. 고양이와 강아지는 포유류이고 개구리는 양서류이기 때문이죠. 

+ `keras.layers.Embedding(input_dim, ooutput_dim)`

```python
model2 = keras.Sequential()

model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()

'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 100, 16)           8000      
                                                                 
 simple_rnn_1 (SimpleRNN)    (None, 8)                 200       
                                                                 
 dense_1 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,209
Trainable params: 8,209
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```

+ 500개 토큰을 크기 16 벡터로 변경하였기 때문에 500x16 크기의 모델 파라미터를 가짐
  + 뉴런이 8개 이기 때문에 16x8 개, 은닉상태의 가중치 8x8 + 8개 절편 -> 8200개 모델파라미터가 있고 Dense층 9개이므로 8209개



## LSTM과  GRU 셀

> simpleRNN 보다는 복잡하지만 훨씬 성능이 뛰어난 순환신경망 모델
>
> 기본 순환층은 시퀀스가 길어질수록 순환되는 은닉 상태에 담긴 정보가 점차 희석되기 때문에 긴 시퀀스를 학습하기 어렵다. 그런 이유로 LSTM과 GRU셀이 제안되었다.



### LSTM(Long Short-Term Memory)

>  장단기 메모리

+ 입력게이트, 삭제게이트, 출력게이트를 추가하여 불필요한 기억을 지우고 기억할 것들을 정하여 긴 시퀀스 입력을 잘 처리하도록 설계

![image](https://user-images.githubusercontent.com/101400894/188547752-9eabd86c-ed10-4e16-ab6a-47ffbb87ccf8.png)



#### 구조 제대로 보기

<img src="https://user-images.githubusercontent.com/101400894/188549131-fce2fbd4-4229-4227-b863-0120fc345135.png" alt="image" style="zoom:50%;" align="left"/><img src="https://user-images.githubusercontent.com/101400894/188549200-475c11e3-5efc-45ec-b5cb-9f851dceba7b.png" alt="image" style="zoom:60%;" />

> t시점의 셀 상태를 *c_t* 로 표현하고 있으며 이전 스텝은 *c_t*−1 로 표현하고 있다. *σ*는 시그모이드 함수이다.
> *W_x*− 은 *x_t* 를 사용하는 각 게이트에서의 가중치이다. -자리에 알파벳이 들어가서 구분을 쉽게 할 예정
> *W_h*− 은 *h_t*−1(은닉상태) 를 사용하는 각 게이트에서의 가중치이다. -자리에 알파벳이 들어가서 구분을 쉽게 할 예정
> *b*− 는 각 게이트에서 사용하는 bias 이다.



+ 입력 게이트 : 현재 정보를 기억하기 위한 게이트

<img src="https://user-images.githubusercontent.com/101400894/188549448-f7710c2a-2138-4fcc-bade-2c0c51132c30.png" alt="image" style="zoom:40%;" align="left"/>

> 현 시점의 x값인 *x_t* 를 입력게이트로 이어지는 가중치 *W_xi* 와 곱하고 이전 시전의 은닉상태에 가중치 *W_hi* 를 곱하고 *b_i* 를 더해서 시그모이드 함수에 통과시킨다. 결과인 *i_t*는 0~1 사이의 값을 갖는다.
>
> *x_t* 를 입력게이트로 이어지는 가중치 *W_xg* 와 곱하고 이전 시전의 은닉상태에 가중치 *W_hg* 를 곱하고 *b_i* 를 더해서 하이퍼볼리탄젠트함수에 통과시킨다. 결과인 *g_t*는 -1~1 값을 갖는다. 이 두 값으로 기억할 정보의 양을 정한다.



+ 삭제 게이트 : 셀 상태에 있는 정보를 제거하는 역할

<img src="https://user-images.githubusercontent.com/101400894/188549729-f2918ad0-6eea-4191-8094-fe31b35227bd.png" alt="image" style="zoom:40%;" align="left"/>

> *x_t* 와 *h_t*−1 가 시그모이드 함수를 지나서 *f_t* 로 출력된다. 시그모이드 함수를 지났으므로 0~1 사이인데, 클수록 정보량을 많이 갖고 있다.



+ 셀 상태

<img src="https://user-images.githubusercontent.com/101400894/188549866-f2ce0218-86ac-416f-bc2d-2e464fdc9452.png" alt="image" style="zoom:40%;" align="left"/>

> *g_t* 와 *i_t*를 원소별 곱을 해주고 *f_t*는 *C_t*−1과 원소곱을 하여 둘을 더해줍니다. 만약에 삭제 게이트에서 0에 수렴하는 값을 내보냈다면 이전 셀의 정보는 거의 들어가지 않는다고 봐야합니다.
> 위 계산을 해줌으로써 t시점에서 셀 상태 계산은 끝납니다.



+ 출력 게이트

<img src="https://user-images.githubusercontent.com/101400894/188550008-936e794f-cb09-4f6f-b755-949516c77940.png" alt="image" style="zoom:40%;" align="left"/>

> *x_t*와 *h_t*−1 가 시그모이드 함수를 지난 결과와 셀 상태가 하이퍼볼릭탄젠트 함수를 지난 결과를 서로 원소별 곱해주어 결과를 낸다. 시그모이드 함수 결과인 *o_t*는 현재 시점의 은닉 상태를 결정하고 하이퍼탄젠트 함수를 지난 결과값은 값이 걸러지는 효과가 발생하여 은닉상태가 된다고 한다.
> 처음엔 이걸 보고 삭제게이트와 같은 작업을 왜 두 번 해주는거지? 라고 생각했지만 각 가중치와 편향이 다르기 때문에 다른 작업이었다.





#### 코드보기

```python
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
from tensorflow import keras

#LSTM 신경망
model = keras.Sequential()

#임베딩을 통해 500개에서 16개로 줄이기
model.add(keras.layers.Embedding(500, 16, input_length=100))
#LSTM사용
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 100, 16)           8000      
                                                                 
 lstm (LSTM)                 (None, 8)                 800       
                                                                 
 dense (Dense)               (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,809
Trainable params: 8,809
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', 
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```



### 순환층에 드롭아웃

>  CNN에서 과대적합을 막기 위해 드롭아웃을 했는데 순환층에서는 자체적으로 드롭아웃 기능을 제공한다고 한다. 매개변수 recurrent_dropout 의 비율을 정해주면 된다. 하지만 이것을 사용하면 GPU 를 사용하여 훈련하지는 못한다고 한다. 30% 드롭아웃 시켜서 훈련시켜보자.

+ 매개변수로 드롭아웃 수치 지정

```python
model2 = keras.Sequential()

model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```



#### 2개의 층을 연결하기

+  return_sequences = True
  + 마지막 순환셀이 아니면 그 이전의 모든 순환셀은 매 타임스텝마다 은닉상태를 출력해줘야 함
  + 모든 타임스텝에 대해 은닉상태를 출력하고 싶을 때 사용
  + 마지막 층 제외 모든 층의 은닉상태는 해당 매개변수를 True로 해줘야함

```
model3 = keras.Sequential()

model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))

model3.summary()
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, 100, 16)           8000      
                                                                 
 lstm_2 (LSTM)               (None, 100, 8)            800       
                                                                 
 lstm_3 (LSTM)               (None, 8)                 544       
                                                                 
 dense_2 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 9,353
Trainable params: 9,353
Non-trainable params: 0
_________________________________________________________________
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model3.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model3.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```



### GRU(Gated Recurrent Unit)

>  게이트 순환 유닛
>
>  GRU는 LSTM와 비슷한 역할을 하지만 구조를 간단하게 만든 모델
>
>  셀상태가 없음
>
>  LSTM은 입력, 삭제, 출력게이트가 존재했지만 GRU는 업데이트와 리셋 게이트만 사용

<img src="https://user-images.githubusercontent.com/101400894/188550968-89a00d76-de35-4c6f-891f-047e4947ac73.png" alt="image" style="zoom:80%;" align="left"/>

#### 모델 자세히 보기

<img src="https://user-images.githubusercontent.com/101400894/188550907-3ef5cf30-65e5-4fa1-b2d8-6504f3139095.png" alt="image" style="zoom:40%;" align="left"/>

> r은 리셋 게이트, z는 업데이트 게이트이다. LSTM과 다른 점은 셀 상태가 없다는 점이다.
> 입력과 이전의 은닉상태에 각각 가중치를 곱하고 bias를 더해서 시그모이드 함수를 통과시켜서 리셋게이트에서는 잊혀질 정보의 양을, 업데이트 게이트에서는 유지할 정보의 양을 계산
>
> 리셋게이트 결과를 하이퍼볼릭탄젠트 함수에 통과시키고 업데이트 게이트에는 1을 빼서 두 결과를 원소별로 곱해줌
>
> 또 한 쪽에서는 업데이트 게이트와 이전의 은닉상태를 원소별로 곱해줌
>
> 이 두 결과를 더해서 은닉상태의 정보를 정함



#### 모델 만들기

```
model4 = keras.Sequential()

model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))

model4.summary()

'''
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_3 (Embedding)     (None, 100, 16)           8000      
                                                                 
 gru (GRU)                   (None, 8)                 624       
                                                                 
 dense_3 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,633
Trainable params: 8,633
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model4.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5', 
                                                save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model4.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
```

