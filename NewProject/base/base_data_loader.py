import numpy as np 
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
###
# torch.utils.data : PyTorch 데이터 로딩 유틸리티. 
# 데이터셋을 배치단위로 나누고, 멀티스레드환경에서 데이터를 로드, 데이터 순서를 섞거나, 데이터 병렬처리할 수 있게 해줌
# default_collate : PyTorch 데이터 로더에서 배치를 만들때 사용하는 기본 함수. 여러 샘플을 하나의 배치로 합칠때 사용
# SubsetRandomSample: 데이터셋의 일부 인덱스를 무작위로 샘플링할때 쓰임
##

class BaseDataLoader(DataLoader):  # PyTorch의 데이터 로더를 상속 받음 => 이후 MNIST데이터 로더에 상속 할 예정
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):

        self.validation_split = validation_split # 훈련 - 검증셋 분리
        self.shuffle = shuffle # 무작위로 샘플링할지의 여부 True / False

        self.batch_idx = 0 
        self.n_samples = len(dataset) # 샘플갯수 => 데이터셋 길이

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        # 훈련/ 검증 샘플을 나눔

        self.init_kwargs = { # DataLoader 초기화에 필요한 인자를 딕셔너리 형태로 저장
            'dataset': dataset,# 데이터셋
            'batch_size': batch_size, # 배치크기
            'shuffle': self.shuffle, #shuffle 여부
            'collate_fn': collate_fn, #데이터를 배치로 합치는 함수 => 여기서는 PyTorch에서 제공하는 함수 사용
            'num_workers': num_workers # 데이터 로딩에 사용하는 서브프로세스 갯수
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs) # 훈련용 샘플러와 함께 넘겨줌

    def _split_sampler(self, split): # 훈련/ 검증 샘플을 나누는 함수
        if split == 0.0: # 0.0 값이면 훈련/검증 샘플을 나누지 않는다.
            return None, None

        idx_full = np.arange(self.n_samples) #데이터셋의 인덱스를 생성

        np.random.seed(0)
        np.random.shuffle(idx_full) # 인덱스 배열을 무작위로 섞음
        

        if isinstance(split, int): # split 값에 따라 데이터셋의 크기 결정
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            # 검증셋 사이즈가 전체 샘플수가 많을때 알려줨

            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        # 검증 및 훌련 데이터 셋 인덱스 분할
        valid_idx = idx_full[0:len_valid] # len_valid의 인덱스를 선택하여 검증 데이터셋으로 사용
        train_idx = np.delete(idx_full, np.arange(0, len_valid)) #검증 데이터 셋 인덱스를 제외한 나머지 인덱스를 선택

        train_sampler = SubsetRandomSampler(train_idx) # 훈련 데이터 셋 인덱스 배열로 무작위로 샘플링
        valid_sampler = SubsetRandomSampler(valid_idx) # 검증 데이터 셋 인덱스 배열로 무작위로 샘플링

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False # 샘플러와 섞이는 것을 방지하기 위함
        self.n_samples = len(train_idx) # 샘플 갯수는 훈련인덱스 개수

        return train_sampler, valid_sampler # 각 분리된 샘플러 반환

    def split_validation(self): # 검증용 데이터로더 반환 
        if self.valid_sampler is None: 
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
