# python ../train.py --method normal
# python ../train.py --method swa --csv_name swa_log.csv

# python ../train.py --method threshold_swa --mode gt --csv_name swa_thresh_gt.csv 
# python ../train.py --method threshold_swa --mode lt --csv_name swa_thresh_lt.csv

# python plot_log.py

# python ../train.py --method threshold_swa --selection_type threshold --update_type masking --mode gt --threshold 0.01 --csv_name thres_gt_0.01.csv --epochs 100
# python ../train.py --method threshold_swa --selection_type threshold --update_type masking --mode gt --threshold 0.05 --csv_name thres_gt_0.05.csv --epochs 100
# python ../train.py --method threshold_swa --selection_type threshold --update_type masking --mode lt --threshold 0.01 --csv_name thres_lt_0.01.csv --epochs 100
# python ../train.py --method threshold_swa --selection_type threshold --update_type masking --mode lt --threshold 0.05 --csv_name thres_lt_0.05.csv --epochs 100
# python ../train.py --method threshold_swa --selection_type topk --update_type masking --topk_ratio 0.9 --csv_name topk_90.csv --epochs 100
# python ../train.py --method threshold_swa --selection_type bottomk --update_type masking --topk_ratio 0.9 --csv_name bottomk_90.csv --epochs 100

python ../train.py --method threshold_swa --selection_type bottomk --update_type masking --topk_ratio 0.0 --csv_name bottomk_0.csv --epochs 200
python ../train.py --method threshold_swa --selection_type bottomk --update_type masking --topk_ratio 0.1 --csv_name bottomk_10.csv --epochs 200
python ../train.py --method threshold_swa --selection_type bottomk --update_type masking --topk_ratio 0.5 --csv_name bottomk_50.csv --epochs 200
python ../train.py --method threshold_swa --selection_type bottomk --update_type masking --topk_ratio 0.9 --csv_name bottomk_90.csv --epochs 200

python ../train.py --method threshold_swa --selection_type topk --update_type masking --topk_ratio 0.0 --csv_name topk_0.csv --epochs 200
python ../train.py --method threshold_swa --selection_type topk --update_type masking --topk_ratio 0.1 --csv_name topk_10.csv --epochs 200
python ../train.py --method threshold_swa --selection_type topk --update_type masking --topk_ratio 0.5 --csv_name topk_50.csv --epochs 200
python ../train.py --method threshold_swa --selection_type topk --update_type masking --topk_ratio 0.9 --csv_name topk_90.csv --epochs 200

# python ../train.py \
#   --method threshold_swa \
#   --threshold 1e-9 \
#   --selection_type threshold \
#   --mode gt \
#   --update_type masking \
#   --topk_ratio 1.0 \
#   --csv_name threshold_swa_loose_mask.csv

# python ../train.py \
#   --method threshold_swa \
#   --threshold 1e-9 \
#   --selection_type threshold \
#   --mode gt \
#   --update_type weighted \
#   --topk_ratio 1.0 \
#   --csv_name threshold_swa_weighted_loose.csv


python plot_log.py
