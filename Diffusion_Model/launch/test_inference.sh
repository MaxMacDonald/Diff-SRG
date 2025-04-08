cd /home/max/mastersProject/Radar-Diffusion-main/diffusion_consistency_radar
export PYTHONPATH=$(pwd)

python scripts/image_sample_radar_custom.py --training_mode edm  --sigma_max 120 --sigma_min 0.002 --s_churn 0 --steps 100 --sampler euler  --attention_resolutions 32,16,8  --class_cond False --dropout 0.02 --image_size 32 --num_channels 64 --num_heads 4 --num_res_blocks 3 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --weight_schedule karras --in_ch 2 --out_ch 1

python /home/max/mastersProject/Midas/visualiseDopplerDataDiffusion.py --input_folder /home/max/Results/generatedData_Diffusion --visualisationFolder /home/max/Results/dopplerVisualisationDiffusion
