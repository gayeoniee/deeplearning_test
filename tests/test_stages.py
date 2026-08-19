"""2단계 구조 회귀 테스트 — 합성 매니페스트로 검증.

여기서 잡으려는 사고 3가지:
  1. 1단계와 2단계가 **서로 다른 분할**을 쓰면 누수가 생깁니다.
     to_stage1/to_stage2 는 fold/is_holdout/group 을 그대로 물려받아야 합니다.
  2. 2단계에 A7(정상)이 한 장이라도 섞이면 "정상"이 7번째 병변이 되어버립니다.
  3. stage1_scores 가 엉뚱한 열을 집으면 임계값 전체가 무의미해집니다.
     CLASSES_STAGE1 = ["A7", "ABNORMAL"] 이므로 '이상' 은 index 1 입니다.

    python -m pytest tests/test_stages.py -q
    또는  python tests/test_stages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import split, stages                                    # noqa: E402
from src.config import CFG, CLASSES, CLASSES_STAGE1, NORMAL_LABEL  # noqa: E402
from src.stages import ABNORMAL_LABEL                            # noqa: E402


def make_manifest(n_animals: int = 120, per_animal: int = 8, seed: int = 0) -> pd.DataFrame:
    """실물(VL01)과 비슷한 비율의 합성 매니페스트.

    실물: A7 22,815(49%) / A2 7,693 / A3 6,720 / A1 4,061 / A4 1,631 / A6 1,501 / A5 1,464
    """
    rng = np.random.default_rng(seed)
    labels = [NORMAL_LABEL] + CLASSES
    probs = np.array([0.49, 0.09, 0.17, 0.15, 0.035, 0.032, 0.033])
    probs = probs / probs.sum()

    rows = []
    for a in range(n_animals):
        # 한 개체의 사진은 같은 라벨을 가지는 경향이 있으므로 그렇게 만듭니다
        lab = rng.choice(labels, p=probs)
        for k in range(per_animal):
            rows.append({
                "image_name": f"IMG_{a:04d}_{k}.jpg",
                "image_path": f"/fake/IMG_{a:04d}_{k}.jpg",
                "crop_path": f"/fake/crop/IMG_{a:04d}_{k}.jpg",
                "label": lab,
                "animal_id": f"G_{a:04d}",
                "phash": f"{a:04d}{k}",
                "src_split": "train",
            })
    return pd.DataFrame(rows)


def test_stage_views_share_one_split():
    df = make_manifest()
    df = split.assign(df, CFG(), verbose=False)

    s1 = stages.to_stage1(df, verbose=False)
    s2 = stages.to_stage2(df, verbose=False)

    # (1) 같은 분할을 쓰는가 — 열이 살아 있고 값이 그대로인가
    for col in ("fold", "is_holdout", "group"):
        assert col in s1.columns and col in s2.columns, f"{col} 이 사라졌습니다"
    assert (s1["fold"].to_numpy() == df["fold"].to_numpy()).all()
    assert (s1["is_holdout"].to_numpy() == df["is_holdout"].to_numpy()).all()

    # (2) 두 단계의 그룹이 원본 분할을 벗어나지 않는가 (= 새 분할을 만들지 않았는가)
    for k in range(CFG().n_folds):
        g_val = set(s2[(~s2["is_holdout"]) & (s2["fold"] == k)]["group"])
        g_tr = set(s2[(~s2["is_holdout"]) & (s2["fold"] != k)]["group"])
        assert not (g_val & g_tr), f"2단계 fold {k} 에서 그룹 누수"

    # (3) 1단계/2단계 각각을 split.verify 로 통과시켜야 합니다
    assert split.verify(s1, fold=0, strict=True)
    assert split.verify(s2, fold=0, strict=True)


def test_stage1_is_binary_and_keeps_original_label():
    df = split.assign(make_manifest(), CFG(), verbose=False)
    s1 = stages.to_stage1(df, verbose=False)

    assert set(s1["label"].unique()) <= set(CLASSES_STAGE1)
    assert len(s1) == len(df), "1단계는 전체 데이터를 씁니다 (한 장도 버리지 않음)"
    # 원래 라벨을 보존해야 2단계로 넘길 때 다시 쓸 수 있습니다
    assert "label_orig" in s1.columns
    assert (s1["label_orig"].to_numpy() == df["label"].to_numpy()).all()
    # 매핑이 맞는가
    was_normal = df["label"] == NORMAL_LABEL
    assert (s1.loc[was_normal, "label"] == NORMAL_LABEL).all()
    assert (s1.loc[~was_normal, "label"] == ABNORMAL_LABEL).all()


def test_stage2_excludes_normal():
    df = split.assign(make_manifest(), CFG(), verbose=False)
    s2 = stages.to_stage2(df, verbose=False)

    assert NORMAL_LABEL not in set(s2["label"].unique()), \
        "정상이 2단계에 남으면 '정상'이 7번째 병변이 됩니다"
    assert set(s2["label"].unique()) <= set(CLASSES)
    assert len(s2) == int((df["label"] != NORMAL_LABEL).sum())


def test_stage1_scores_picks_abnormal_column():
    # ABNORMAL 이 확실한 logit — index 1 이 커야 합니다
    logits = np.array([[5.0, -5.0],    # 정상 확신
                       [-5.0, 5.0]])   # 이상 확신
    s = stages.stage1_scores(logits)
    assert s[0] < 0.05, "정상 샘플의 '이상 확률' 이 높게 나옴 → 열을 잘못 집었습니다"
    assert s[1] > 0.95


def test_binary_targets_mapping():
    i_no = CLASSES_STAGE1.index(NORMAL_LABEL)
    i_ab = CLASSES_STAGE1.index(ABNORMAL_LABEL)
    y = np.array([i_no, i_ab, i_ab, i_no])
    assert (stages.binary_targets(y) == np.array([0, 1, 1, 0])).all()


def test_report_pipeline_penalizes_stage1_misses():
    rng = np.random.default_rng(0)
    n = 400
    s1_true = rng.integers(0, 2, n)
    # 완벽한 1단계 vs 절반만 잡는 1단계
    good = np.where(s1_true == 1, 0.9, 0.1)
    bad = np.where(s1_true == 1, rng.choice([0.9, 0.1], n), 0.1)

    s2_true = rng.integers(0, len(CLASSES), 200)
    s2_logits = np.eye(len(CLASSES))[s2_true] * 8.0     # 2단계는 완벽

    r_good = stages.report_pipeline(good, s1_true, s2_logits, s2_true, 0.5, verbose=False)
    r_bad = stages.report_pipeline(bad, s1_true, s2_logits, s2_true, 0.5, verbose=False)

    assert r_good["stage1_recall"] == 1.0
    assert r_good["stage1_missed"] == 0
    assert r_good["stage2_macro_f1"] > 0.99
    # 2단계가 완벽해도 1단계가 놓치면 파이프라인 점수는 떨어져야 합니다
    assert r_bad["pipeline_estimate"] < r_good["pipeline_estimate"] - 0.2
    assert r_bad["stage1_missed"] > 0


def _fake_two_stage(y_labels, s1_recall=1.0, s2_correct=1.0, seed=0):
    """정답 라벨 배열로부터 (s1_scores, s2_logits) 를 합성합니다."""
    rng = np.random.default_rng(seed)
    y_labels = np.asarray(y_labels)
    is_lesion = y_labels != NORMAL_LABEL

    # 1단계: 병변이면 높은 점수를 주되 (1-s1_recall) 만큼은 놓치게
    hit = rng.random(len(y_labels)) < s1_recall
    s1 = np.where(is_lesion & hit, 0.9, 0.1)

    # 2단계: 정답 클래스에 큰 logit, (1-s2_correct) 만큼은 엉뚱한 클래스로
    ok = rng.random(len(y_labels)) < s2_correct
    tgt = []
    for lab, good in zip(y_labels, ok):
        i = CLASSES.index(lab) if lab in CLASSES else 0
        if not good:
            i = (i + 1) % len(CLASSES)
        tgt.append(i)
    s2 = np.eye(len(CLASSES))[np.array(tgt)] * 8.0
    return s1, s2


def test_pipeline_predict_routes_correctly():
    y = np.array([NORMAL_LABEL, "A2", "A5"])
    s1, s2 = _fake_two_stage(y)
    pred, conf = stages.pipeline_predict(s1, s2, threshold=0.5)

    i_normal = len(CLASSES)
    assert pred[0] == i_normal, "1단계 미달인데 병변으로 갔습니다"
    assert stages.PIPELINE_CLASSES[pred[1]] == "A2"
    assert stages.PIPELINE_CLASSES[pred[2]] == "A5"
    assert len(conf) == 3 and (conf >= 0).all() and (conf <= 1).all()


def test_pipeline_predict_rejects_misaligned_inputs():
    s1 = np.array([0.9, 0.9, 0.9])
    s2 = np.eye(len(CLASSES))[[0, 1]]        # 행 2개 — 어긋남
    try:
        stages.pipeline_predict(s1, s2, 0.5)
    except ValueError as e:
        assert "행 수가 다릅니다" in str(e)
    else:
        raise AssertionError("어긋난 입력을 그냥 통과시켰습니다 — 조용히 틀린 점수가 나옵니다")


def test_pipeline_report_counts_missed_lesions():
    y = np.array([NORMAL_LABEL] * 100 + ["A2"] * 50 + ["A5"] * 50)

    perfect = stages.pipeline_report(*_fake_two_stage(y), y, 0.5, show=False)
    assert perfect["lesion_missed"] == 0
    assert perfect["lesion_screening_recall"] == 1.0
    assert perfect["false_alarm_rate"] == 0.0
    assert perfect["final_macro_f1"] > 0.99

    leaky = stages.pipeline_report(*_fake_two_stage(y, s1_recall=0.6, seed=1), y, 0.5, show=False)
    assert leaky["lesion_missed"] > 0
    assert leaky["lesion_screening_recall"] < 0.8
    # 놓친 병변은 전부 A7 로 예측됐어야 합니다 → A7 의 precision 이 떨어집니다
    i_normal = stages.PIPELINE_CLASSES.index(NORMAL_LABEL)
    assert leaky["per_class"]["precision"][i_normal] < 1.0


def test_pipeline_report_separates_screening_from_kind():
    """1단계는 완벽하고 2단계만 틀리면: 놓친 병변 0, 종류 정확도만 하락."""
    y = np.array([NORMAL_LABEL] * 60 + ["A1"] * 40 + ["A3"] * 40)
    rep = stages.pipeline_report(*_fake_two_stage(y, s2_correct=0.5, seed=2), y, 0.5, show=False)

    assert rep["lesion_missed"] == 0, "1단계가 완벽한데 병변을 놓쳤다고 나옵니다"
    assert rep["lesion_screening_recall"] == 1.0
    assert 0.3 < rep["kind_accuracy_given_routed"] < 0.75
    # 스크리닝은 멀쩡한데 최종 macro-F1 은 떨어져야 합니다 — 두 문제가 분리돼야 함
    assert rep["final_macro_f1"] < 0.9


def test_pipeline_report_rejects_unknown_label():
    y = np.array(["A2", "고양이"])
    s1, s2 = _fake_two_stage(np.array(["A2", "A2"]))
    try:
        stages.pipeline_report(s1, s2, y, 0.5, show=False)
    except ValueError as e:
        assert "모르는 라벨" in str(e)
    else:
        raise AssertionError("모르는 라벨을 통과시켰습니다")


def test_threshold_tradeoff_direction():
    """임계값을 낮추면 놓친 병변은 줄고 오탐은 늘어야 합니다."""
    rng = np.random.default_rng(7)
    y = np.array([NORMAL_LABEL] * 300 + ["A2"] * 300)
    is_lesion = y != NORMAL_LABEL
    # 겹치는 분포 — 임계값이 실제로 의미를 갖게
    s1 = np.where(is_lesion, rng.normal(0.65, 0.2, len(y)), rng.normal(0.35, 0.2, len(y)))
    s1 = s1.clip(0, 1)
    s2 = np.eye(len(CLASSES))[np.full(len(y), CLASSES.index("A2"))] * 8.0

    low = stages.pipeline_report(s1, s2, y, 0.3, show=False)
    high = stages.pipeline_report(s1, s2, y, 0.7, show=False)

    assert low["lesion_missed"] < high["lesion_missed"]
    assert low["false_alarm_rate"] > high["false_alarm_rate"]


def test_stage1_is_roughly_balanced_on_realistic_mix():
    """실물 비율(정상 49%)에서 1단계가 5:5 근처로 나오는지."""
    df = make_manifest(n_animals=400, per_animal=6, seed=3)
    s1 = stages.to_stage1(df, verbose=False)
    frac = (s1["label"] == NORMAL_LABEL).mean()
    assert 0.35 < frac < 0.65, f"정상 비율 {frac:.2f} — 1단계가 심하게 불균형합니다"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
