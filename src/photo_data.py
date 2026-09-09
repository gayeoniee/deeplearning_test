"""고정 ROI 를 고정해 두고 **`photometric` 만** 켜고 끄는 복원 확인 (이력 감사 ①).

STEP 44·45 파일럿에는 채택 레시피의 `photometric` 이 **빠져 있었습니다**.
그래서 *"safe 가 졌다"* 가 crop 탓인지 레시피 탓인지 안 갈립니다.

여기서는 **두 팔 다 고정 ROI** 로 두고 학습 증강만 다르게 합니다 —
바뀌는 것이 하나뿐이라 그 하나의 효과가 나옵니다.
⚠️ LR·EMA·기간까지 같이 바꾸면 다시 못 가릅니다.

연산자와 확률은 `src.data` 의 `photometric` 프리셋과 **같습니다**."""
import numpy as np
from PIL import Image
from src.roi_data import ROIDataset


class PhotoNormalize:
    def __init__(self, normalize, seed):
        import albumentations as A
        self.normalize = normalize
        # 연산자·확률을 src.data 의 photometric 프리셋과 똑같이 맞춥니다.
        self.aug = A.Compose([
            A.CLAHE(clip_limit=2.0, p=.2),
            A.OneOf([A.GaussianBlur(blur_limit=(3,7)),
                     A.MotionBlur(blur_limit=(3,7))], p=.3),
            A.GaussNoise(p=.25),
            A.ImageCompression(quality_range=(40,90), p=.3),
        ], seed=seed)
        self.worker = None

    def __call__(self, image):
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        if info is not None and self.worker != info.id:
            self.aug.set_random_seed(info.seed % (2**32))
            self.worker = info.id
        return self.normalize(Image.fromarray(self.aug(image=np.asarray(image))['image']))


class PhotoDataset(ROIDataset):
    def __init__(self, *args, mode='fixed', **kwargs):
        if mode not in ('fixed','photo'):
            raise ValueError(mode)
        super().__init__(*args, mode='fixed', **kwargs)
        if self.train and mode == 'photo':
            self.normalize = PhotoNormalize(self.normalize, self.cfg.seed + self.cfg.photo_epoch)
