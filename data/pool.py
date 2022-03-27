import os
from logging import getLogger
from tqdm import tqdm
import torch
import torch.nn.functional as F


class PJFPool(object):
    def __init__(self, config):
        self.logger = getLogger()
        self.config = config

        self._load_ids()

    def _load_ids(self):
        for target in ['geek', 'job']:
            token2id = {}
            id2token = []
            filepath = os.path.join(self.config['dataset_path'], f'{target}.token')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r') as file:
                for i, line in enumerate(file):
                    token = line.strip()
                    token2id[token] = i
                    id2token.append(token)
            setattr(self, f'{target}_token2id', token2id)
            setattr(self, f'{target}_id2token', id2token)
            setattr(self, f'{target}_num', len(id2token))

    def __str__(self):
        return '\n\t'.join(['Pool:'] + [
            f'{self.geek_num} geeks',
            f'{self.job_num} jobs'
        ])

    def __repr__(self):
        return self.__str__()


class TextPool(PJFPool):
    def __init__(self, config):
        super(TextPool, self).__init__(config)
        self._load_word_cnt()
        self._load_longsent()

    def _load_word_cnt(self):
        min_word_cnt = self.config['min_word_cnt']
        self.wd2id = {
            '[WD_PAD]': 0,
            '[WD_MISS]': 1
        }
        self.id2wd = ['[WD_PAD]', '[WD_MISS]']
        filepath = os.path.join(self.config['dataset_path'], 'word.cnt')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                wd, cnt = line.strip().split('\t')
                if int(cnt) < min_word_cnt:
                    break
                self.wd2id[wd] = i + 2
                self.id2wd.append(wd)
        self.wd_num = len(self.id2wd)

    def _load_longsent(self):
        for target in ['geek', 'job']:
            max_sent_len = self.config[f'{target}_longsent_len']
            id2longsent = torch.zeros([getattr(self, f'{target}_num'), max_sent_len], dtype=torch.int64)
            id2longsent_len = torch.zeros(getattr(self, f'{target}_num'))
            filepath = os.path.join(self.config['dataset_path'], f'{target}.longsent')
            token2id = getattr(self, f'{target}_token2id')
            self.logger.info(f'Loading {filepath}')
            with open(filepath, 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    try:
                        token, longsent = line.strip().split('\t')
                    except:
                        token = line.strip()
                        longsent = '[WD_PAD]'
                    if token not in token2id:
                        continue
                    idx = token2id[token]
                    longsent = torch.LongTensor([self.wd2id[_] if _ in self.wd2id else 1 for _ in longsent.split(' ')])
                    id2longsent[idx] = F.pad(longsent, (0, max_sent_len - longsent.shape[0]))
                    id2longsent_len[idx] = min(max_sent_len, longsent.shape[0])
            setattr(self, f'{target}_id2longsent', id2longsent)
            setattr(self, f'{target}_id2longsent_len', id2longsent_len)

    def __str__(self):
        return '\n\t'.join([
            super(TextPool, self).__str__(),
            f'{self.wd_num} words',
            f'geek_id2longsent: {self.geek_id2longsent.shape}',
            f'job_id2longsent: {self.job_id2longsent.shape}'
        ])


class SHPJFPool(TextPool):
    def __init__(self, config):
        super(SHPJFPool, self).__init__(config)

    def _load_ids(self):
        super(SHPJFPool, self)._load_ids()
        filepath = os.path.join(self.config['dataset_path'], 'job.search.token')
        self.logger.info(f'Loading {filepath}')
        ori_job_num = self.job_num
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                token = line.strip()
                assert token not in self.job_token2id
                self.job_token2id[token] = i + ori_job_num
                self.job_id2token.append(token)
        self.job_search_token_num = len(self.job_id2token) - ori_job_num
        self.job_num = len(self.job_id2token)

    def _load_word_cnt(self):
        super(SHPJFPool, self)._load_word_cnt()
        ori_wd_num = len(self.id2wd)
        filepath = os.path.join(self.config['dataset_path'], 'word.search.id')
        self.logger.info(f'Loading {filepath}')
        with open(filepath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                wd = line.strip()
                assert wd not in self.wd2id
                self.wd2id[wd] = i + ori_wd_num
                self.id2wd.append(wd)
        self.search_wd_num = len(self.id2wd) - ori_wd_num
        self.wd_num = len(self.id2wd)

    def __str__(self):
        return '\n\t'.join([
            super(SHPJFPool, self).__str__(),
            f'{self.job_search_token_num} job tokens only exist in search log',
            f'{self.search_wd_num} words only exist in search log'
        ])
