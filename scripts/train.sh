# python ../train.py --method normal
# python ../train.py --method swa --csv_name swa_log.csv

# 1. threshold × masking × gt
python ../train.py --method threshold_swa \
  --threshold median \
  --selection_type threshold \
  --update_type masking \
  --mode gt \
  --csv_name thres_mask_gt.csv \
  --epochs 10

# 2. threshold × masking × lt
python ../train.py --method threshold_swa \
  --threshold median \
  --selection_type threshold \
  --update_type masking \
  --mode lt \
  --csv_name thres_mask_lt.csv \
  --epochs 10

# 3. threshold × weighted × gt
python ../train.py --method threshold_swa \
  --threshold median \
  --selection_type threshold \
  --update_type weighted \
  --mode gt \
  --csv_name thres_weight_gt.csv \
  --epochs 10

# 4. threshold × weighted × lt
python ../train.py --method threshold_swa \
  --threshold median \
  --selection_type threshold \
  --update_type weighted \
  --mode lt \
  --csv_name thres_weight_lt.csv \
  --epochs 10

# 5. topk × masking
python ../train.py --method threshold_swa \
  --selection_type topk \
  --topk_ratio 0.5 \
  --update_type masking \
  --csv_name topk_mask.csv \
  --epochs 10

# 6. topk × weighted
python ../train.py --method threshold_swa \
  --selection_type topk \
  --topk_ratio 0.5 \
  --update_type weighted \
  --csv_name topk_weight.csv \
  --epochs 10

# 7. bottomk × masking
python ../train.py --method threshold_swa \
  --selection_type bottomk \
  --topk_ratio 0.5 \
  --update_type masking \
  --csv_name bottomk_mask.csv \
  --epochs 10

# 8. bottomk × weighted
python ../train.py --method threshold_swa \
  --selection_type bottomk \
  --topk_ratio 0.5 \
  --update_type weighted \
  --csv_name bottomk_weight.csv \
  --epochs 10


python plot_log.py