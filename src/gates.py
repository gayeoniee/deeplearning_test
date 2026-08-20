"""품질 게이트 — 멈춰야 할 때와 알려야 할 때를 나눕니다.

⚠️ **왜 노트북이 아니라 여기 있나**

노트북 셀은 `git pull` 로 갱신되지 않습니다. Colab/Kaggle 에 한 번 올린 `.ipynb`
는 다시 import 하기 전까지 그대로입니다. 반면 `src/` 는 첫 셀이 매번 당겨옵니다.

그래서 **바뀔 수 있는 판단 기준은 전부 여기 둡니다.** 실제로 겪은 일:
1단계 정지선을 0.80 → 0.70 으로 고쳐 푸시했는데, 사용자의 Kaggle 노트북은
낡은 셀을 들고 있어서 AUROC 0.798 에서 그대로 멈췄습니다.

기준선 두 종류:
    STOP  — 파이프라인이 **깨진** 수준. 여기서만 멈춥니다.
    WANT  — 기대치. 못 넘으면 경고하되 진행합니다.

    "기대보다 낮다" 와 "깨졌다" 는 다른 문제입니다.
    무인 실행(Kaggle Save & Run All)에서 전자로 멈추면 몇 시간이 날아갑니다.
"""

from __future__ import annotations

# 1단계: 정상/이상 이진 판정 (AUROC)
#   랜덤 = 0.50, 실측 = 0.8031 / 0.8100
STAGE1_STOP = 0.70
STAGE1_WANT = 0.80

# 2단계: 병변 6종 (macro-F1)
#   랜덤 = 1/6 ≈ 0.167, 실측 = 0.4865
STAGE2_STOP = 0.20
STAGE2_WANT = 0.25

# 파이프라인: 스크리닝 재현율 — 놓친 병변이 가장 위험한 오류
PIPELINE_RECALL_WANT = 0.95


def stage1(auroc: float, threshold: float, precision: float,
           floor: float | None = None, baseline: float | None = None,
           crop_tag: str = "?", epochs: int | None = None) -> bool:
    """1단계 게이트. 통과 여부를 돌려주고, **깨진 수준일 때만** 예외를 냅니다."""
    if auroc <= STAGE1_STOP:
        raise AssertionError(
            f"1단계 AUROC {auroc:.3f} — 정상/이상을 거의 구분하지 못합니다 "
            f"(랜덤 0.50, 정지선 {STAGE1_STOP}).\n"
            "모델을 바꾸기 전에 크롭과 라벨을 다시 확인하세요 (노트북 1번)."
        )

    ok = auroc >= STAGE1_WANT
    if ok:
        print(f"✅ 1단계 통과 — AUROC {auroc:.4f}, 임계값 {threshold:.4f}")
    else:
        print(f"⚠️ 1단계 AUROC {auroc:.4f} — 목표 {STAGE1_WANT} 에 못 미칩니다 "
              f"(정지선 {STAGE1_STOP} 은 넘었으므로 계속 진행).")
        print("   이 숫자를 그대로 보고하지 마세요. 원인 후보:")
        if crop_tag != "full":
            print(f"   · 1단계 크롭이 '{crop_tag}' 입니다 — ROI 크롭은 배율로 정답을 흘립니다")
        else:
            print("   · 입력 해상도 — full 크롭에서 병변은 20~40px 뿐입니다")
        print("   · 학습 곡선에서 val loss 가 바닥을 쳤는지 (덜 학습됐을 수 있음)")
        print("   · 하한선(사진 안 보고 맞히기) 대비 격차 — 아래 숫자를 보세요")

    if precision < 0.5:
        print(f"⚠️ precision {precision:.3f} — 알림 절반 이상이 헛알림입니다.")

    if floor is not None:
        print(f"\n  사진을 안 본 하한선 : {floor:.4f}   (크롭 '{crop_tag}' 에서는 참고용)")
        print(f"  CNN                : {auroc:.4f}")
    if baseline is not None:
        ep = f", {epochs}ep" if epochs else ""
        print(f"\n  첫 실측              : {baseline:.4f}")
        print(f"  이번({crop_tag}{ep})".ljust(23) + f": {auroc:.4f}   ({auroc - baseline:+.4f})")
    return ok


def stage2(macro_f1: float, floor: float | None = None,
           baseline: float | None = None) -> bool:
    """2단계 게이트."""
    if macro_f1 <= STAGE2_STOP:
        raise AssertionError(
            f"2단계 macro-F1 {macro_f1:.3f} — 랜덤(0.167) 수준입니다 "
            f"(정지선 {STAGE2_STOP}).\n"
            "모델을 바꾸지 말고 크롭·라벨을 다시 확인하세요."
        )

    ok = macro_f1 >= STAGE2_WANT
    if ok:
        print(f"✅ 2단계 게이트 통과 — macro-F1 {macro_f1:.4f}")
    else:
        print(f"⚠️ macro-F1 {macro_f1:.4f} — 랜덤(0.167)보다 조금 나은 수준입니다. "
              "결과를 신뢰하지 마세요.")

    if floor is not None:
        lift = macro_f1 - floor
        print(f"\n[하한선 대비]  사진 안 봄 {floor:.4f} → CNN {macro_f1:.4f}  ({lift:+.4f})")
        if lift < 0.05:
            print("  🚨 거의 못 넘었습니다 — 크롭 배율을 보고 있을 수 있습니다.")
        elif lift < 0.15:
            print("  ⚠️ 격차가 작습니다. 6번 배율 교란 검사를 꼭 보세요.")
        else:
            print("  ✅ 피부에서 실제로 배웠습니다.")
    if baseline is not None:
        print(f"\n  첫 실측 : {baseline:.4f}  →  이번 : {macro_f1:.4f}  "
              f"({macro_f1 - baseline:+.4f})")
    return ok
