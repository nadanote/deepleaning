# 코드 2-1
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 훈련세트: 모델이 훈련해야 할 것// 테스트 세트

print(train_images.shape) # (60000, 28, 28) # [0,255] 사이의 값인 unit8타입의 (60000, 28,28) 크기를 가진 배열
print(len(train_labels)) # 60000
print(train_labels) # [5 0 4 ... 5 6 8]
print(train_labels.dtype) # uint8

# test data
print(test_images.shape) # (10000, 28, 28)
print(len(test_labels))  # 10000
print(test_labels.dtype) # uint8


'''
작업 순서
1. 훈련 데이터 train_images와 train_labels를 네트워크에 주입
   -> 네트워크는 이미지와 레이블을 연관시킬 수 있도록 학습됨
   
2. test_images에 대한 예측을 네트어크에 요청
   -> 이 예측이 test_labels와 맞는지 확인
'''
