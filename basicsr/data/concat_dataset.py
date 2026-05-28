import copy
from torch.utils import data as data
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ConcatLoViFDataset(data.Dataset):
    """
    Concatenate multiple LoViF track datasets into one big dataset.
    Each batch randomly samples from any of the 5 tracks.

    Usage in YAML train config:
        datasets:
          train:
            name: LoViF_AllTracks
            type: ConcatLoViFDataset
            tracks:
              Blur:
                gt: /datasets_host/LoViF 2026/Blur/GT
                lq: /datasets_host/LoViF 2026/Blur/LQ
              Haze:
                gt: /datasets_host/LoViF 2026/Haze/GT
                lq: /datasets_host/LoViF 2026/Haze/LQ
              Lowlight:
                gt: /datasets_host/LoViF 2026/Lowlight/GT
                lq: /datasets_host/LoViF 2026/Lowlight/LQ
              Rain:
                gt: /datasets_host/LoViF 2026/Rain/GT
                lq: /datasets_host/LoViF 2026/Rain/LQ
              Snow:
                gt: /datasets_host/LoViF 2026/Snow/GT
                lq: /datasets_host/LoViF 2026/Snow/LQ
            gt_size: 64
            use_hflip: true
            use_rot: true
            scale: 1
            io_backend: {type: disk}
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.tracks = opt['tracks']

        self.datasets = []
        self.cum_sizes = [0]
        for name, paths in self.tracks.items():
            sub_opt = copy.deepcopy(opt)
            sub_opt['dataroot_gt'] = paths['gt']
            sub_opt['dataroot_lq'] = paths['lq']
            sub_opt['name'] = f'{opt["name"]}_{name}'
            ds = PairedImageDataset(sub_opt)
            self.datasets.append(ds)
            self.cum_sizes.append(self.cum_sizes[-1] + len(ds))

        self.total = self.cum_sizes[-1]
        print(f'[ConcatLoViFDataset] {len(self.tracks)} tracks, {self.total} total images')

    def __getitem__(self, index):
        for i in range(len(self.datasets)):
            if index < self.cum_sizes[i + 1]:
                local_idx = index - self.cum_sizes[i]
                return self.datasets[i][local_idx]
        raise IndexError

    def __len__(self):
        return self.total
