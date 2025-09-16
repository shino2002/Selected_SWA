# -------- Weighted + Threshold (gt) --------
python ../train.py \
  --method threshold_swa \
  --selection_type threshold \
  --threshold 0.01 \
  --mode gt \
  --update_type weighted \
  --csv_name weighted_thresh_gt_001.csv \
  --epochs 50

python ../train.py \
  --method threshold_swa \
  --selection_type threshold \
  --threshold 0.05 \
  --mode gt \
  --update_type weighted \
  --csv_name weighted_thresh_gt_005.csv \
  --epochs 50

# -------- Weighted + Threshold (lt) --------
python ../train.py \
  --method threshold_swa \
  --selection_type threshold \
  --threshold 0.01 \
  --mode lt \
  --update_type weighted \
  --csv_name weighted_thresh_lt_001.csv \
  --epochs 50

python ../train.py \
  --method threshold_swa \
  --selection_type threshold \
  --threshold 0.05 \
  --mode lt \
  --update_type weighted \
  --csv_name weighted_thresh_lt_005.csv \
  --epochs 50

# -------- Weighted + TopK --------
python ../train.py \
  --method threshold_swa \
  --selection_type topk \
  --topk_ratio 0.1 \
  --update_type weighted \
  --csv_name weighted_topk_10.csv \
  --epochs 50

python ../train.py \
  --method threshold_swa \
  --selection_type topk \
  --topk_ratio 0.01 \
  --update_type weighted \
  --csv_name weighted_topk_01.csv \
  --epochs 50

  