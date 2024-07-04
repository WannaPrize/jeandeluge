import torch.nn as nn  # 파이토치 신경망 모델 구축을 위한 nn 모듈
import numpy as np
from abc import abstractmethod # 추상 메서드를 정의하는데쓰임


class BaseModel(nn.Module): #nn.Module을 상속받음
    """
    Base class for all models
    """
    @abstractmethod # 추상 메서드임을 나타내는 데코레이터 => 하위 에서는 꼭 이 메서드를 재정의하여 구현해야함
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError #추상 메서드임을 알리고, 이 부분이 정의되지 않았을경우 에러를 만듦.

    def __str__(self): #객체를 문자열로 표현할 때 호출되는 특수 매서드
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters()) # 모델 학습 가능한 매개변수 필터링 prquires_grad = True인 매개변수만 선택됨
        params = sum([np.prod(p.size()) for p in model_parameters]) #매개 면수의 크기를 곱하여 총 수를 계산
        return super().__str__() + '\nTrainable parameters: {}'.format(params) #학습 가능한 매개변 수를 문자열에서 보여줌
