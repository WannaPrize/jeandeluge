from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # data_dir : 데이터 위치
        # batch 사이즈
        # shuffle= True: 무작위로 샘플링 함 True
        # valication_split = 훈련/  검증 샘플을 나누지 않고, 샘플러를 반환하지 않음
        # num_worker : 병렬 처리 프로세스 수 여기서는 1개
        # training True 인 경우 훈련 데이터 셋 사용한다은 의미
        trsfm = transforms.Compose([ # 이미지 
            transforms.ToTensor(), # 이미지를 텐서로 변환
            transforms.Normalize((0.1307,), (0.3081,)) #이미지 정규화 , MNIST는 평균이 0.1307, 표준편차가 0.3081
        ])
        self.data_dir = data_dir # 인스턴스 변수로 데이터 디렉토리 경로 저장
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        # PyTorch MNIST 데이터 로드. 훈렷셋 설정. transform에 trsfm정의한 변환 적용
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        # BASEDATALOADER 생성자 호출하여 초기화
