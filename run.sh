######### train DeepGrid 4x3 ##########
python train.py --batch_size 1 \
--epochs 100 \
--BCE_weight 1.0 \
--IOU_weight 1.0 \
--DSC_weight 1.0 \
--lr 0.00001 \
--lr_red1 50 \
--lr_red2 100 \
--lr_red_factor 10 \
--dataset isic2017 \
--dataset_path ISIC_2017 \
--model_save_dir logmods \
--model_load_path logmods \
--log_path logmods/logs.json
