export PYTHONPATH="${PYTHONPATH}:/home/parhamsa/projects/def-arbeltal/parhamsa/LSGM"
DATA_DIR=/home/parhamsa/projects/def-arbeltal/parhamsa/LSGM/data
CHECKPOINT_DIR=/home/parhamsa/projects/def-arbeltal/parhamsa/LSGM/checkpoints
FID_STATS_DIR=/home/parhamsa/projects/def-arbeltal/parhamsa/LSGM/fid_stats
EXPR_ID=000

python train_vada.py --data $DATA_DIR/mnist --root $CHECKPOINT_DIR --save $EXPR_ID/lsgm --dataset mnist --epochs 1 \
        --dropout 0.2 --batch_size 32 --num_scales_dae 2 --weight_decay_norm_vae 1e-2 \
        --weight_decay_norm_dae 0. --num_channels_dae 256 --train_vae  --num_cell_per_scale_dae 8 \
        --learning_rate_dae 3e-4 --learning_rate_min_dae 3e-4 --train_ode_solver_tol 1e-5 --cont_kl_anneal  \
        --sde_type vpsde --iw_sample_p ll_iw --num_process_per_node 1 --use_se \
        --vae_checkpoint $CHECKPOINT_DIR/$EXPR_ID/vae/checkpoint.pt  --dae_arch ncsnpp --embedding_scale 1000 \
        --mixing_logit_init -6 --warmup_epochs 20 --drop_inactive_var --skip_final_eval --fid_dir $FID_STATS_DIR