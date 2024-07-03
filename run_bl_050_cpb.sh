python AE_baseline.py 	-c  ./hyperparameters/ae_cp_050_512_cpb.yaml \
		--epochs        700 \
            --lr            1.00e-04 \
            --cpb           200 \
            --seed          4321 \
            --runmode       baseline \
            --wandb         \
            --gpu_id        1 \
            # --run_id   	qy61mxif \
            # --ckpt     	AE_base_scpb200-050Ltnt_512_20240617_ep_450.pt
