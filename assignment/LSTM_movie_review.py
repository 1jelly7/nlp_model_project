# -*- coding: utf-8 -*-
"""
한국어 영화 리뷰 감성 분석 LSTM 모델 - PyTorch Lightning 버전
ratings-data.txt 파일을 읽어 학습하는 예제입니다.
"""

# ---------------------------------------------------------------------
# 1. 기본 라이브러리 불러오기
# ---------------------------------------------------------------------

# csv는 탭(구분자) 기반의 텍스트 파일을 열로 나누어 읽을 때 사용합니다.
import csv

# random은 데이터 순서를 섞거나 재현 가능한 분할을 할 때 사용합니다.
import random

# re는 정규표현식을 사용하여 텍스트 전처리를 할 때 사용합니다.
import re

# Counter는 단어 빈도를 세어 vocabulary를 만들 때 사용합니다.
from collections import Counter

# dataclass는 설정값을 하나의 객체로 관리하기 좋게 묶어 줍니다.
from dataclasses import dataclass

# Path는 운영체제에 관계없이 파일 경로를 안전하게 다루기 위해 사용합니다.
from pathlib import Path

# typing은 함수 인자와 반환값의 타입을 명시하기 위해 사용합니다.
from typing import Dict, List, Tuple

# 원본 문장으로 캐시 키를 만들기 위해 사용합니다.
import hashlib

# 토큰 캐시 파일을 JSON 형식으로 저장하고 읽기 위해 사용합니다.
import json

# 형태소 분석을 프로세스 병렬 처리하기 위해 사용합니다.
from concurrent.futures import ProcessPoolExecutor

# 한국어 형태소 분석기 Okt를 사용합니다.
from konlpy.tag import Okt

# ---------------------------------------------------------------------
# 2. 딥러닝 라이브러리 불러오기
# ---------------------------------------------------------------------

# torch는 PyTorch의 핵심 라이브러리입니다.
import torch

# nn은 Embedding, LSTM, Linear, Dropout 같은 신경망 계층을 제공합니다.
import torch.nn as nn

# Dataset과 DataLoader는 데이터를 배치 단위로 모델에 공급할 때 사용합니다.
from torch.utils.data import DataLoader, Dataset

# PyTorch Lightning은 학습 루프를 구조적으로 관리하기 쉽게 해 줍니다.
import pytorch_lightning as pl

# BinaryAccuracy는 이진 분류 정확도를 계산할 때 사용합니다.
from torchmetrics.classification import BinaryAccuracy

# ---------------------------------------------------------------------
# 3. 설정값 정의
# ---------------------------------------------------------------------

@dataclass
class Config:
    """프로젝트 전체 설정값을 저장하는 클래스입니다."""

    # 학습에 사용할 한국어 리뷰 데이터 파일 경로입니다.
    # 실제 첨부 파일명은 ratings-data.txt 입니다.
    data_path: str = '../data/ratings.txt'

    # 한 문장에서 사용할 최대 토큰 수입니다.
    # 긴 문장은 앞에서부터 max_len개만 사용하고 짧은 문장은 패딩합니다.
    max_len: int = 80

    # vocabulary에 포함할 최대 단어 수입니다.
    # 너무 크게 잡으면 메모리 사용량이 증가할 수 있습니다.
    max_vocab_size: int = 30000

    # vocabulary에 포함되기 위한 최소 등장 빈도입니다.
    # 2보다 작게 등장한 단어는 제외합니다.
    min_freq: int = 2

    # 한 번의 학습 단계에서 사용할 샘플 수입니다.
    batch_size: int = 128

    # 단어 하나를 몇 차원의 벡터로 표현할지 지정합니다.
    embedding_dim: int = 128

    # LSTM 은닉 상태 벡터의 차원 수입니다.
    hidden_dim: int = 128

    # LSTM 층 개수입니다.
    num_layers: int = 1

    # 과적합을 줄이기 위한 Dropout 비율입니다.
    dropout: float = 0.3

    # 옵티마이저의 학습률입니다.
    learning_rate: float = 3e-4

    # 전체 데이터를 몇 번 반복 학습할지 지정합니다.
    max_epochs: int = 5

    # 전체 데이터 중 검증 데이터 비율입니다.
    val_ratio: float = 0.1

    # 전체 데이터 중 테스트 데이터 비율입니다.
    test_ratio: float = 0.1

    # DataLoader 병렬 worker 수입니다.
    # Windows/PyCharm 환경에서는 0이 가장 안전합니다.
    num_workers: int = 0

    # 재현 가능한 실행을 위한 랜덤 시드입니다.
    seed: int = 42

    # 형태소 분석 결과를 저장할 캐시 파일 경로입니다.
    morph_cache_path: str = './ratings_morph_cache.json'

    # 형태소 분석 병렬 처리에 사용할 프로세스 수입니다.
    morph_num_workers: int = 4


# ---------------------------------------------------------------------
# 4. 텍스트 전처리 함수
# ---------------------------------------------------------------------

# 각 프로세스에서 재사용할 전역 Okt 객체입니다.
_okt = None


def _get_okt():
    """프로세스별로 Okt 객체를 한 번만 생성해서 재사용합니다."""

    # 전역 Okt 객체를 사용합니다.
    global _okt

    # 아직 생성되지 않은 경우에만 초기화합니다.
    if _okt is None:
        _okt = Okt()

    return _okt


def _tokenize_worker(text: str) -> List[str]:
    """병렬 프로세스에서 문장 하나를 형태소 분석하는 함수입니다."""

    # 현재 프로세스에서 사용할 Okt 객체를 가져옵니다.
    okt = _get_okt()

    # 정규화와 어간 추출을 적용해 품사 분석을 수행합니다.
    pos_tokens = okt.pos(text, norm=True, stem=True)

    # 제거할 조사, 어미, 문장부호 품사를 정의합니다.
    stop_pos = {'Josa', 'Eomi', 'Punctuation'}

    # 불필요 품사를 제거하고 짧은 감성 핵심 토큰은 예외로 유지합니다.
    tokens = [word for word, pos in pos_tokens if pos not in stop_pos and (len(word) > 1 or word in {'안', '못'})]

    # 최종 토큰 리스트를 반환합니다.
    return tokens


def make_cache_key(text: str) -> str:
    """문장별 캐시 조회를 위한 해시 키를 생성합니다."""

    # 동일 문장이면 항상 같은 키가 생성되도록 SHA-256 해시를 사용합니다.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """한국어 리뷰 문장을 모델 입력용으로 정리합니다."""

    # 혹시 숫자나 None이 들어오더라도 문자열로 안전하게 변환합니다.
    text = str(text)

    # HTML 태그가 있다면 공백으로 바꿉니다.
    text = re.sub(r"<.*?>", " ", text)

    # 한글, 영문, 숫자, 자모, 기본 문장부호를 제외한 문자는 공백으로 바꿉니다.
    text = re.sub(r"[^가-힣a-zA-Z0-9ㄱ-ㅎㅏ-ㅣ!?.,' ]", " ", text)

    # 여러 개의 공백을 하나의 공백으로 줄입니다.
    text = re.sub(r"\s+", " ", text)

    # 영문은 소문자로 통일하고 양끝 공백을 제거합니다.
    text = text.lower().strip()

    # 정리된 텍스트를 반환합니다.
    return text


def tokenize(text: str) -> List[str]:
    """문장을 형태소 분석하여 토큰 리스트로 분리합니다."""

    # 형태소 분석 전에 텍스트를 정리합니다.
    text = clean_text(text)

    # Okt 기반 형태소 분석 결과를 반환합니다.
    return _tokenize_worker(text)


# ---------------------------------------------------------------------
# 5. ratings.txt 파일 로드 함수
# ---------------------------------------------------------------------

def read_ratings_file(data_path: Path, config: Config) -> List[Tuple[List[str], int]]:
    """ratings-data.txt 파일에서 리뷰와 라벨을 읽어옵니다."""

    # 파일이 실제로 존재하는지 먼저 확인합니다.
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    # 파일에서 읽은 원본 텍스트와 라벨을 임시 저장합니다.
    raw_rows = []

    # UTF-8 인코딩으로 파일을 엽니다.
    with open(data_path, 'r', encoding='utf-8') as f:

        # 탭 구분 파일이므로 delimiter를 '\t'로 지정합니다.
        reader = csv.DictReader(f, delimiter='\t')

        # 반드시 있어야 하는 컬럼 이름입니다.
        required_columns = {'document', 'label'}

        # 헤더가 없거나 필요한 컬럼이 빠져 있으면 오류를 발생시킵니다.
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"파일 형식이 올바르지 않습니다. 필요한 컬럼: {required_columns}, 실제 컬럼: {reader.fieldnames}"
            )

        # 파일의 각 행을 하나씩 읽습니다.
        for row in reader:
            # 리뷰 문장을 가져와 전처리합니다.
            text = clean_text(row['document'])

            # 전처리 결과가 비어 있으면 학습에 도움이 되지 않으므로 건너뜁니다.
            if not text:
                continue

            # 라벨 값을 정수형으로 변환합니다.
            label = int(row['label'])

            # 라벨이 0 또는 1인 경우만 사용합니다.
            if label not in (0, 1):
                continue

            # (문장, 라벨) 형태로 샘플 리스트에 추가합니다.
            raw_rows.append((text, label))

    # 캐시 파일 경로를 Path 객체로 변환합니다.
    cache_path = Path(config.morph_cache_path)

    # 이전 실행에서 저장한 캐시 파일이 있는지 확인합니다.
    if cache_path.exists():
        # 캐시 파일을 엽니다.
        with open(cache_path, 'r', encoding='utf-8') as f:
            # 캐시 JSON을 딕셔너리로 읽어옵니다.
            morph_cache = json.load(f)
    # 캐시 파일이 없으면 빈 딕셔너리로 시작합니다.
    else:
        # 새 캐시 저장소를 초기화합니다.
        morph_cache = {}

    # 아직 캐시에 없는 문장만 따로 모읍니다.
    missing_texts = []
    # 캐시에 저장할 키 목록을 함께 유지합니다.
    missing_keys = []

    # 전체 문장을 순회하며 캐시 존재 여부를 확인합니다.
    for text, _ in raw_rows:
        # 현재 문장의 캐시 키를 생성합니다.
        key = make_cache_key(text)
        # 캐시에 없는 문장만 병렬 처리 대상으로 분류합니다.
        if key not in morph_cache:
            # 저장할 캐시 키를 기록합니다.
            missing_keys.append(key)
            # 형태소 분석이 필요한 문장을 기록합니다.
            missing_texts.append(text)

    # 새로 분석할 문장이 하나라도 있으면 병렬 형태소 분석을 수행합니다.
    if missing_texts:
        # 병렬 처리 대상 수를 출력합니다.
        print(f"[형태소 분석] 새 문장 {len(missing_texts)}개를 병렬 처리합니다.")
        # 프로세스 풀을 생성합니다.
        with ProcessPoolExecutor(max_workers=config.morph_num_workers) as executor:
            # 문장 리스트를 병렬 형태소 분석합니다.
            tokenized_results = list(executor.map(_tokenize_worker, missing_texts))
        # 분석 결과를 캐시 딕셔너리에 반영합니다.
        for key, tokens in zip(missing_keys, tokenized_results):
            # 문장 해시 키 기준으로 토큰 리스트를 저장합니다.
            morph_cache[key] = tokens
        # 갱신된 캐시를 파일에 저장합니다.
        with open(cache_path, 'w', encoding='utf-8') as f:
            # 한글이 깨지지 않도록 캐시를 JSON으로 저장합니다.
            json.dump(morph_cache, f, ensure_ascii=False)
        # 캐시 저장 완료 메시지를 출력합니다.
        print(f"[형태소 분석] 캐시 저장 완료: {cache_path}")
    # 모든 문장이 캐시에 있으면 형태소 분석을 생략합니다.
    else:
        # 캐시 재사용 메시지를 출력합니다.
        print(f"[형태소 분석] 기존 캐시 사용: {cache_path}")

    # 최종적으로 (토큰 리스트, 라벨) 쌍을 저장할 리스트입니다.
    samples: List[Tuple[List[str], int]] = []

    # 원본 문장 순서대로 최종 샘플을 만듭니다.
    for text, label in raw_rows:
        # 현재 문장의 캐시 키를 다시 생성합니다.
        key = make_cache_key(text)
        # 캐시에서 토큰 리스트를 가져옵니다.
        tokens = morph_cache[key]
        # 토큰이 비어 있으면 학습에 도움이 되지 않으므로 제외합니다.
        if not tokens:
            # 다음 샘플로 넘어갑니다.
            continue
        # 토큰 리스트와 라벨 쌍을 저장합니다.
        samples.append((tokens, label))

    # 라벨이나 입력 순서 편향을 줄이기 위해 데이터를 섞습니다.
    random.shuffle(samples)

    # 최종 샘플 리스트를 반환합니다.
    return samples


def split_samples(
    samples: List[Tuple[List[str], int]],
    val_ratio: float,
    test_ratio: float
) -> Tuple[List[Tuple[List[str], int]], List[Tuple[List[str], int]], List[Tuple[List[str], int]]]:
    """전체 샘플을 train/val/test로 분할합니다."""

    # 전체 샘플 개수를 계산합니다.
    total_size = len(samples)

    # 테스트 데이터 개수를 계산합니다.
    test_size = int(total_size * test_ratio)

    # 검증 데이터 개수를 계산합니다.
    val_size = int(total_size * val_ratio)

    # 나머지를 훈련 데이터로 사용합니다.
    train_size = total_size - val_size - test_size

    # 훈련 데이터 개수가 0 이하이면 비율 설정이 잘못된 것입니다.
    if train_size <= 0:
        raise ValueError("train/val/test 분할 비율이 잘못되었습니다.")

    # 앞부분은 훈련 데이터로 사용합니다.
    train_samples = samples[:train_size]

    # 중간 부분은 검증 데이터로 사용합니다.
    val_samples = samples[train_size:train_size + val_size]

    # 마지막 부분은 테스트 데이터로 사용합니다.
    test_samples = samples[train_size + val_size:]

    # 분할된 세 개의 리스트를 반환합니다.
    return train_samples, val_samples, test_samples


def load_data(config: Config) -> Tuple[List[Tuple[List[str], int]], List[Tuple[List[str], int]], List[Tuple[List[str], int]]]:
    """데이터 파일을 읽고 train/val/test로 나누어 반환합니다."""

    # 설정에 지정된 데이터 경로를 Path 객체로 변환합니다.
    data_path = Path(config.data_path)

    # TSV 파일에서 전체 샘플을 읽어옵니다.
    samples = read_ratings_file(data_path, config)

    # 전체 샘플을 train/val/test로 나눕니다.
    train_samples, val_samples, test_samples = split_samples(
        samples, config.val_ratio, config.test_ratio
    )

    # 분할 결과를 출력하여 데이터가 정상적으로 로드되었는지 확인합니다.
    print(
        f"[데이터 로드 완료] total={len(samples)}, "
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    # 훈련/검증/테스트 데이터를 반환합니다.
    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------
# 6. Vocabulary 생성 함수
# ---------------------------------------------------------------------

def build_vocab(samples: List[Tuple[List[str], int]], config: Config) -> Dict[str, int]:
    """훈련 데이터에서 vocabulary를 생성합니다."""

    # 단어 빈도를 저장할 Counter 객체를 만듭니다.
    counter = Counter()

    # 모든 훈련 샘플을 순회합니다.
    for tokens, _ in samples:
        # 토큰 리스트로 빈도를 누적합니다.
        counter.update(tokens)

    # 특수 토큰을 먼저 vocabulary에 등록합니다.
    # <PAD>는 패딩용, <UNK>는 vocabulary에 없는 단어용입니다.
    word_to_index: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}

    # 빈도가 높은 단어부터 최대 크기까지 vocabulary에 추가합니다.
    for word, freq in counter.most_common(config.max_vocab_size - len(word_to_index)):

        # 최소 빈도보다 작은 단어는 제외합니다.
        if freq < config.min_freq:
            continue

        # 아직 vocabulary에 없는 단어만 등록합니다.
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

    # 최종 vocabulary 크기를 출력합니다.
    print(f"[Vocabulary 생성 완료] 단어 수: {len(word_to_index)}")

    # 완성된 vocabulary를 반환합니다.
    return word_to_index


def encode_text(tokens: List[str], word_to_index: Dict[str, int], max_len: int) -> torch.Tensor:
    """토큰 리스트를 고정 길이 정수 텐서로 변환합니다."""

    # 각 토큰을 vocabulary 인덱스로 바꿉니다.
    # vocabulary에 없는 단어는 <UNK> 인덱스를 사용합니다.
    token_ids = [word_to_index.get(token, word_to_index['<UNK>']) for token in tokens]

    # 너무 긴 문장은 max_len까지만 사용합니다.
    token_ids = token_ids[:max_len]

    # 너무 짧은 문장은 뒤에 <PAD>를 채워 길이를 맞춥니다.
    if len(token_ids) < max_len:
        token_ids = token_ids + [word_to_index['<PAD>']] * (max_len - len(token_ids))

    # 정수 리스트를 torch.long 타입 텐서로 변환합니다.
    return torch.tensor(token_ids, dtype=torch.long)


# ---------------------------------------------------------------------
# 7. Dataset 클래스 정의
# ---------------------------------------------------------------------

class RatingsDataset(Dataset):
    """한국어 리뷰 데이터와 라벨을 제공하는 Dataset 클래스입니다."""

    def __init__(self, samples: List[Tuple[List[str], int]], word_to_index: Dict[str, int], max_len: int):
        # 원본 샘플 리스트를 저장합니다.
        self.samples = samples

        # 단어를 인덱스로 바꾸기 위한 vocabulary를 저장합니다.
        self.word_to_index = word_to_index

        # 문장 최대 길이를 저장합니다.
        self.max_len = max_len

    def __len__(self) -> int:
        # 전체 샘플 개수를 반환합니다.
        return len(self.samples)

    def __getitem__(self, index: int):
        # 지정된 위치의 텍스트와 라벨을 가져옵니다.
        tokens, label = self.samples[index]

        # 텍스트를 고정 길이 정수 텐서로 변환합니다.
        input_ids = encode_text(tokens, self.word_to_index, self.max_len)

        # 라벨을 LongTensor로 변환합니다.
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 모델 입력과 정답 라벨을 함께 반환합니다.
        return input_ids, label_tensor


# ---------------------------------------------------------------------
# 8. LightningDataModule 정의
# ---------------------------------------------------------------------

class RatingsDataModule(pl.LightningDataModule):
    """데이터 준비와 DataLoader 생성을 담당하는 DataModule입니다."""

    def __init__(self, config: Config):
        # 부모 클래스 초기화입니다.
        super().__init__()

        # 설정 객체를 멤버 변수로 저장합니다.
        self.config = config

        # setup()에서 생성할 vocabulary를 저장할 변수입니다.
        self.word_to_index: Dict[str, int] = {}

        # setup()에서 생성할 Dataset 객체들입니다.
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # setup()이 중복 실행되는 것을 막기 위한 플래그입니다.
        self._is_setup = False

    def prepare_data(self) -> None:
        # 한 번만 수행되는 다운로드 작업 등이 있을 때 사용하는 메서드입니다.
        # 현재 예제에서는 별도 다운로드가 없으므로 비워 둡니다.
        pass

    def setup(self, stage: str = None) -> None:
        # 이미 한 번 setup이 끝났으면 다시 실행하지 않습니다.
        if self._is_setup:
            return

        # 데이터 파일을 읽고 train/val/test로 분리합니다.
        train_samples, val_samples, test_samples = load_data(self.config)

        # 훈련 데이터만 사용하여 vocabulary를 생성합니다.
        self.word_to_index = build_vocab(train_samples, self.config)

        # 훈련 Dataset 객체를 생성합니다.
        self.train_dataset = RatingsDataset(train_samples, self.word_to_index, self.config.max_len)

        # 검증 Dataset 객체를 생성합니다.
        self.val_dataset = RatingsDataset(val_samples, self.word_to_index, self.config.max_len)

        # 테스트 Dataset 객체를 생성합니다.
        self.test_dataset = RatingsDataset(test_samples, self.word_to_index, self.config.max_len)

        # setup 완료 상태를 기록합니다.
        self._is_setup = True

        # Dataset 준비 결과를 출력합니다.
        print(
            f"[Dataset 준비 완료] "
            f"train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        # 훈련용 DataLoader를 생성합니다.
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        # 검증용 DataLoader를 생성합니다.
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        # 테스트용 DataLoader를 생성합니다.
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


# ---------------------------------------------------------------------
# 9. LSTM 모델 정의
# ---------------------------------------------------------------------

class LSTMClassifier(pl.LightningModule):
    """한국어 영화 리뷰 감성 분석을 위한 LSTM 분류 모델입니다."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        pad_index: int = 0,
    ):
        # LightningModule 초기화입니다.
        super().__init__()

        # 하이퍼파라미터를 체크포인트 등에 저장할 수 있도록 기록합니다.
        self.save_hyperparameters()

        # 학습률을 멤버 변수로 저장합니다.
        self.learning_rate = learning_rate

        # Embedding 계층은 단어 인덱스를 dense vector로 변환합니다.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        # LSTM 계층은 단어 순서를 고려하여 문장 의미를 학습합니다.
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Dropout 계층은 과적합을 줄이는 역할을 합니다.
        self.dropout = nn.Dropout(dropout)

        # 최종 분류 계층은 hidden_dim을 2개 클래스 점수로 변환합니다.
        # bidirectional=True면 양방향 hidden을 이어붙이므로 입력 차원을 2배로 늘립니다.
        self.classifier = nn.Linear(hidden_dim * 2, 2)

        # 손실 함수로 CrossEntropyLoss를 사용합니다.
        self.loss_fn = nn.CrossEntropyLoss()

        # 훈련 정확도를 계산하는 객체입니다.
        self.train_acc = BinaryAccuracy()

        # 검증 정확도를 계산하는 객체입니다.
        self.val_acc = BinaryAccuracy()

        # 테스트 정확도를 계산하는 객체입니다.
        self.test_acc = BinaryAccuracy()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 패딩이 아닌 실제 토큰의 길이를 계산합니다.
        lengths = (input_ids != 0).sum(dim=1).cpu()
        # 길이가 0인 경우(전부 PAD) 최소 1로 보정합니다.
        lengths = lengths.clamp(min=1)

        # 입력 토큰 인덱스를 임베딩 벡터로 변환합니다.
        embedded = self.embedding(input_ids)

        # PAD를 제외하고 LSTM이 실제 토큰만 처리하도록 패킹합니다.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM에 임베딩 시퀀스를 입력합니다.
        _, (hidden, _) = self.lstm(packed)

        # 양방향 마지막 hidden state를 합칩니다.
        sentence_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)

        # 문장 벡터에 Dropout을 적용합니다.
        sentence_vector = self.dropout(sentence_vector)

        # 문장 벡터를 2개 클래스의 logits로 변환합니다.
        logits = self.classifier(sentence_vector)

        # softmax 이전의 logits를 반환합니다.
        return logits

    def _shared_step(self, batch, stage: str):
        # 배치에서 입력 텐서와 라벨 텐서를 꺼냅니다.
        input_ids, labels = batch

        # 모델의 예측 logits를 계산합니다.
        logits = self(input_ids)

        # 예측 logits와 정답 라벨로 손실을 계산합니다.
        loss = self.loss_fn(logits, labels)

        # 가장 점수가 높은 클래스를 예측값으로 선택합니다.
        preds = torch.argmax(logits, dim=1)

        # 단계별로 사용할 정확도 계산 객체를 선택합니다.
        if stage == "train":
            acc = self.train_acc(preds, labels)
        elif stage == "val":
            acc = self.val_acc(preds, labels)
        else:
            acc = self.test_acc(preds, labels)

        # 손실 값을 로그에 기록합니다.
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # 정확도 값을 로그에 기록합니다.
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # 학습 시에는 loss가 역전파에 사용됩니다.
        return loss

    def training_step(self, batch, batch_idx):
        # 훈련 배치 하나에 대한 손실을 반환합니다.
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # 검증 배치 하나에 대한 지표를 계산합니다.
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        # 테스트 배치 하나에 대한 지표를 계산합니다.
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Adam optimizer를 생성합니다.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # 검증 손실이 정체되면 학습률을 자동으로 낮춥니다.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # 생성한 optimizer와 scheduler를 반환합니다.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }


# ---------------------------------------------------------------------
# 10. 예측 함수
# ---------------------------------------------------------------------

def predict_sentiment(
    model: LSTMClassifier,
    text: str,
    word_to_index: Dict[str, int],
    config: Config
) -> Tuple[str, float]:
    """학습된 모델로 문장 하나의 감성을 예측합니다."""

    # 모델을 평가 모드로 전환합니다.
    model.eval()

    # 예측 과정에서는 그래디언트를 계산할 필요가 없으므로 no_grad를 사용합니다.
    with torch.no_grad():
        # 예측 문장을 형태소 분석하여 토큰 리스트로 변환합니다.
        tokens = tokenize(text)

        # 토큰 리스트를 인덱스 텐서로 변환합니다.
        input_ids = encode_text(tokens, word_to_index, config.max_len)

        # 배치 차원을 추가하여 형태를 (1, 문장길이)로 만듭니다.
        input_ids = input_ids.unsqueeze(0)

        # 모델이 올라가 있는 장치와 같은 장치로 입력 텐서를 이동합니다.
        input_ids = input_ids.to(model.device)

        # 모델의 예측 logits를 계산합니다.
        logits = model(input_ids)

        # softmax를 적용하여 클래스별 확률로 변환합니다.
        probabilities = torch.softmax(logits, dim=1)

        # 가장 높은 확률을 가진 클래스 번호를 선택합니다.
        pred_id = torch.argmax(probabilities, dim=1).item()

        # 해당 클래스의 확률값을 신뢰도로 사용합니다.
        confidence = probabilities[0, pred_id].item()

        # 숫자 라벨을 사람이 읽기 쉬운 문자열로 변환합니다.
        label = "positive" if pred_id == 1 else "negative"

        # 예측 라벨과 신뢰도를 반환합니다.
        return label, confidence


# ---------------------------------------------------------------------
# 11. main 함수
# ---------------------------------------------------------------------

def main() -> None:
    """전체 실행 흐름을 담당하는 main 함수입니다."""

    # 설정 객체를 생성합니다.
    config = Config()

    # 실행 결과 재현성을 위해 랜덤 시드를 고정합니다.
    pl.seed_everything(config.seed, workers=True)

    # DataModule 객체를 생성합니다.
    data_module = RatingsDataModule(config)

    # setup()을 먼저 실행하여 vocabulary와 Dataset을 준비합니다.
    data_module.setup(stage="fit")

    # vocabulary 크기를 구합니다.
    vocab_size = len(data_module.word_to_index)

    batch = next(iter(data_module.train_dataloader()))  # 추가: 첫 배치를 가져와 입력 인덱스를 점검합니다.
    input_ids, labels = batch  # 추가: 입력과 라벨을 분리합니다.

    # 모델 객체를 생성합니다.
    model = LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        pad_index=data_module.word_to_index['<PAD>'],
    )

    # GPU 사용 가능 여부에 따라 accelerator를 자동 선택합니다.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # PyTorch Lightning Trainer를 생성합니다.
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )

    # 모델 학습을 시작합니다.
    trainer.fit(model, datamodule=data_module)

    # 테스트 데이터로 최종 성능을 확인합니다.
    trainer.test(model, datamodule=data_module)

    # 학습 후 예측을 확인할 예시 문장들입니다.
    examples = [
        "정말 재미있고 감동적인 영화였다",
        "시간이 너무 아깝고 지루한 영화였다",
    ]

    # 예측 예시 결과를 출력합니다.
    print("\n[예측 예시]")
    for text in examples:

        # 각 문장에 대해 감성 예측을 수행합니다.
        label, confidence = predict_sentiment(model, text, data_module.word_to_index, config)

        # 원문 문장을 출력합니다.
        print(f"문장: {text}")

        # 예측 라벨과 신뢰도를 출력합니다.
        print(f"예측: {label}, 신뢰도: {confidence:.4f}\n")


# ---------------------------------------------------------------------
# 12. 프로그램 시작 지점
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # 스크립트를 직접 실행한 경우에만 main() 함수를 호출합니다.
    main()