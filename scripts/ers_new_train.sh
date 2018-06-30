$GYM_PYTHON -m baselines.ers.addpg_solver \
	--env=$ENV \
	--seed=0 \
	--test_seed=42 \
	--ob_dtype=float32 \
	--nstack=3 \
	--nn_size="[128,96]" \
	--softmax_actor=True \
	--log_transform_inputs=True \
	--tau=0.001 \
	--exploration_episodes=10 \
	--use_param_noise=True \
	--use_safe_noise=True \
	--exploration_theta=1 \
	--training_episodes=20000 \
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
	--run_no_prefix=ddpg_nips5_seed_0
# --render=True \
# --test_mode=True \
# --saved_model=$OPENAI_LOGDIR/$ENV/ddpg_new_000/model
# nips was batch norm for obs
# nips2 was log norm for both
# nips3 was abs for gamma
# nips4 was epsilon inside log and also fixed adaptive noise scale param
# nips5 is with shift and scale
