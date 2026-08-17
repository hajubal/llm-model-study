#!/usr/bin/env python
"""과제 1·2 정답 · 누수를 두 종류 만들어 검사기의 한계를 드러낸다.

먼저 직접 만들어 본 뒤 비교할 것.

    --mode group   : group_id를 무시하고 랜덤 행 분할한다 -> inspect_data.py가 **잡는다**
    --mode dup     : 같은 문장에 다른 group_id를 붙여 split을 가른다 -> 검사기를 **통과한다**
    --mode near    : 문장 끝에 공백/구두점만 더해 near-duplicate를 만든다 -> 역시 통과한다

핵심은 세 번째다. 검사기는 `group_id` 겹침과 정확히 같은 id만 본다. **텍스트가 같거나
거의 같은지는 보지 않는다.** 그래서 "검사 통과"는 "누수 없음"이 아니라 "우리가 검사한
종류의 누수는 없음"일 뿐이다.

near-duplicate가 더 어려운 이유: 무엇을 '거의 같다'고 볼지 기준이 필요하다. 편집 거리?
정규화 후 일치? 임베딩 유사도? 기준마다 잡는 것이 다르고, 느슨하면 정상 데이터를 지우고
빡빡하면 누수를 놓친다. 정답이 하나가 아니라서 어렵다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from guardlab.io import read_jsonl, write_jsonl
from guardlab.schema import Sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="정상적으로 분할된 데이터 폴더")
    parser.add_argument("--out", required=True, help="누수가 있는 데이터를 쓸 폴더")
    parser.add_argument("--mode", choices=["group", "dup", "near"], default="group")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=20, help="dup/near에서 복제할 샘플 수")
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    splits = {name: read_jsonl(source / f"{name}.jsonl") for name in ("train", "dev", "test")}
    rng = random.Random(args.seed)

    if args.mode == "group":
        # group_id를 완전히 무시하고 전부 섞어 원래 크기대로 다시 자른다.
        # 같은 템플릿에서 나온 문장이 train과 test에 동시에 들어간다.
        everything = [row for rows in splits.values() for row in rows]
        rng.shuffle(everything)
        sizes = {name: len(rows) for name, rows in splits.items()}
        cursor = 0
        leaked = {}
        for name in ("train", "dev", "test"):
            leaked[name] = everything[cursor : cursor + sizes[name]]
            cursor += sizes[name]
        note = "group_id를 무시한 랜덤 행 분할"
    else:
        # test 샘플을 골라 train에 복제한다. group_id를 바꿔 검사기를 통과시킨다.
        leaked = {name: list(rows) for name, rows in splits.items()}
        picked = rng.sample(splits["test"], min(args.n, len(splits["test"])))
        for index, row in enumerate(picked):
            text = row.text if args.mode == "dup" else row.text + " "
            leaked["train"].append(Sample(
                id=f"leak-{args.mode}-{index:03d}",  # id는 유일하므로 중복 검사도 통과한다
                text=text,
                label=row.label,
                source=row.source,
                language=row.language,
                group_id=f"leak-{args.mode}-{index:03d}",  # group_id도 새로 만들어 겹침을 피한다
                meta={"leak_of": row.id, "mode": args.mode},
            ))
        rng.shuffle(leaked["train"])
        note = (
            "test 문장을 train에 그대로 복제(다른 group_id)"
            if args.mode == "dup"
            else "test 문장 끝에 공백만 더해 train에 복제(near-duplicate)"
        )

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in leaked.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    print(f"{note} -> {out}")
    print(f"  train {len(leaked['train'])} / dev {len(leaked['dev'])} / test {len(leaked['test'])}")
    print(f"\n검사기를 돌려 본다:\n  python 02-intermediate/01-dataset-schema/inspect_data.py {out}")
    if args.mode == "group":
        print("  -> group_id 겹침이 보고된다. 검사기가 잡는 종류의 누수다.")
    else:
        print(
            "  -> 통과한다. 그러나 test 문장이 train에 있으므로 test 점수는 부풀려진다.\n"
            "     학습 후 test macro F1이 비정상적으로 높으면 이것을 의심한다."
        )


if __name__ == "__main__":
    main()
