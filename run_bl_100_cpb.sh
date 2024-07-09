python AE_batch.py 	 --config ./hyperparameters/ae_cp_100_512_cpb.yaml \
			--epochs        700 \
                        --lr            1.00e-04 \
                        --cpb           200 \
                        --seed          4321 \
			--runmode       baseline \
                        --wandb         \
			--gpu_id        0 \
                        # --run_id      esy9knk4 \
                        # --ckpt        AE_baseline_20240617_scpb200-100Ltnt_512__ep_450.pt
