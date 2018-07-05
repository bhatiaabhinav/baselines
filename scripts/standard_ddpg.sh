$GYM_PYTHON -m baselines.ers.addpg_solver \
    --env=$1 \
    --ob_dtype=float32 \
    --nstack=3 \
    --nn_size="[64,64]" \
    --tau=0.001 \
    --exploration_episodes=10 \
    --use_param_noise=True \
    --use_safe_noise=False \
    --exploration_sigma=0.2 \
    --exploration_theta=1 \
    --training_episodes=10000 \
    --mb_size=128 \
    --init_scale=3e-3 \
    --lr=1e-3 \
    --a_lr=1e-4 \
    --l2_reg=1e-2 \
    --train_every=2 \
    --exploit_every=4 \
    --logger_level=INFO \
    --use_batch_norm=False \
    --use_layer_norm=True \
    --run_no_prefix=ddpg \
    # --render=True \
    # --test_mode=True \
    # --saved_model=$OPENAI_LOGDIR/$1/ddpg_002/model
