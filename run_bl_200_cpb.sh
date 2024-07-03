python AE_baseline.py   -c  ./hyperparameters/ae_cp_200_512_cpb.yaml \
			--epochs        700 \
                        --lr            1.00e-04 \
                        --cpb           200 \
                        --seed          4321 \
	                --runmode       baseline \
                        --wandb         \
	                --gpu_id        2 \
	                    # --run_id      lgvww2cn \
	                    # --ckpt        AE_baseline_20240617_scpb200-200Ltnt_512__ep_450.pt
