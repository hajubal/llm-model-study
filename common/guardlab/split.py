from __future__ import annotations

import random
from collections import defaultdict

from .schema import LABELS, Sample


def group_stratified_split(
    samples: list[Sample], seed: int = 42, train_ratio: float = 0.7, dev_ratio: float = 0.15,
) -> dict[str, list[Sample]]:
    """라벨별 group_id를 섞고 그룹 전체를 하나의 split에 배치한다."""
    if not 0 < train_ratio < 1 or not 0 <= dev_ratio < 1 or train_ratio + dev_ratio >= 1:
        raise ValueError("split ratio가 올바르지 않습니다")
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    # label과 language를 함께 층화한다. 하나의 그룹에 여러 값이 섞인 경우 첫 샘플의 값을 쓰지 않고 실패시켜
    # 잘못된 group_id 설계를 데이터 생성 단계에서 드러낸다.
    buckets: dict[tuple[str, str], list[list[Sample]]] = defaultdict(list)
    for group_id, rows in groups.items():
        keys = {(row.label, row.language) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"group_id {group_id!r} 안에 label/language가 섞여 있습니다: {sorted(keys)}")
        buckets[next(iter(keys))].append(rows)

    result: dict[str, list[Sample]] = {"train": [], "dev": [], "test": []}
    rng = random.Random(seed)
    for (label, language), bucket_groups in sorted(buckets.items()):
        rng.shuffle(bucket_groups)
        n = len(bucket_groups)
        if n < 3:
            raise ValueError(
                f"{label}/{language}: group이 {n}개뿐입니다. 각 split에 하나씩 두려면 최소 3개가 필요합니다"
            )
        n_train = max(1, min(n - 2, round(n * train_ratio)))
        n_dev = max(1, min(n - n_train - 1, round(n * dev_ratio)))
        for idx, group in enumerate(bucket_groups):
            split = "train" if idx < n_train else "dev" if idx < n_train + n_dev else "test"
            result[split].extend(group)

    for split in result:
        rng.shuffle(result[split])
    assert_no_group_leakage(result)
    return result


def assert_no_group_leakage(splits: dict[str, list[Sample]]) -> None:
    owners: dict[str, str] = {}
    for split, samples in splits.items():
        for sample in samples:
            previous = owners.setdefault(sample.group_id, split)
            if previous != split:
                raise ValueError(f"group_id {sample.group_id!r}가 {previous}/{split}에 중복되었습니다")
