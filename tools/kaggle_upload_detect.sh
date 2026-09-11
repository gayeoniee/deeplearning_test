#!/usr/bin/env bash
# STEP 50 검출 데이터셋 3조각을 캐글 **비공개** Dataset 으로 병렬 업로드합니다.
#
#   1) 캐글 → Settings → API → Generate New Token → 문자열을 ~/.kaggle/access_token 에 저장 (chmod 600)
#   2) bash tools/kaggle_upload_detect.sh
#
# 조각마다 `kaggle datasets create -r zip` 을 동시에 띄웁니다 (images/ 를 zip 으로 묶어 올리고
# 캐글이 풀어 둡니다 — kagglehub 로 받으면 풀린 폴더가 옵니다). 로그는 조각별로 남깁니다.
set -euo pipefail
ROOT=/Users/gayeon/deeplearning_test/data/work/detect_full
[ -f "$HOME/.kaggle/access_token" ] || { echo "~/.kaggle/access_token 이 없습니다 (토큰 문자열 저장)"; exit 1; }
chmod 600 "$HOME/.kaggle/access_token"
cd "$ROOT"
for k in 0 1 2; do
  [ -f "shard$k/dataset-metadata.json" ] || { echo "shard$k/dataset-metadata.json 없음"; exit 1; }
  [ -f "shard$k/boxes.parquet" ] || { echo "shard$k/boxes.parquet 없음 — 추출이 안 끝났습니다"; exit 1; }
  ( uv --directory /Users/gayeon/orca/workspaces/deeplearning_test/bristlemouth run --with 'kaggle>=1.8' \
      kaggle datasets create -p "$ROOT/shard$k" -r zip > "upload_shard$k.log" 2>&1 \
    && echo "shard$k ✅" || echo "shard$k ❌ (upload_shard$k.log)" ) &
done
wait
echo "끝. 캐글 Datasets 페이지에서 세 개가 Private 인지 확인하세요."
