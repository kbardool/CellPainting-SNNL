python AE_baseline.py 	-c  ./hyperparameters/ae_cp_100_512.yaml \
                        --runmode  baseline \
			--wandb    \
			--epochs   200 \
			--gpu_id   1 \
                        --run_id   268fqkf3 \
                        --ckpt     AE_baseline_20240613_snglOpt-100Ltnt_512_ep_450.pt
