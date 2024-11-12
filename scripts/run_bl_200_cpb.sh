python AE_batch.py   \
	        --runmode       baseline \
            --config        ./hyperparameters/ae_cp_200_512_cpb.yaml \
			--epochs        800 \
            --prim_opt      \
            --lr            1.00e-04 \
            --cpb           200 \
            --seed          4321 \
            --wandb         \
	        --gpu_id        2 \
	        # --run_id      lgvww2cn \
	        # --ckpt        AE_baseline_20240617_scpb200-200Ltnt_512_ep_450.pt
