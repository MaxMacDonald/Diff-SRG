cd /home/max/mastersProject/Radar-Diffusion-main/diffusion_consistency_radar
export PYTHONPATH=$(pwd)

python scripts/edm_train_radar_custom.py --lr_anneal_steps 3000 --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 256 --image_size 32 --lr 0.00005 --num_channels 64 --num_head_channels 64  --num_res_blocks 3 --resblock_updown True --schedule_sampler lognormal --use_fp16 False --weight_decay 0.0 --weight_schedule karras --in_ch 2 --out_ch 1
