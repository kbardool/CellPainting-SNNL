python AE_baseline.py 	-c  ./hyperparameters/ae_cp_100_512_cpb200.yaml \
                        --runmode  baseline \
	                --wandb    \
	                --epochs   250 \
	                --gpu_id   0 \
                        --run_id   esy9knk4 \
                        --ckpt     AE_baseline_20240617_scpb200-100Ltnt_512__ep_450.pt
