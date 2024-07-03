	python AE_baseline.py 	-c  ./hyperparameters/ae_sn_200_512_cpb.yaml \
            --runmode       snnl \
            --epochs        200 \
            --temp_opt      --prim_opt \
            --lr            1.00e-04 \
            --temp_lr       1.00e-05 \
            --temp          0.5 \
	        --cpb           200 \
            --seed          4321 \
            --wandb         \
            --gpu_id        0 \
            # --run_id      qw52e4d2 \
            # --ckpt        AE_snnl_dcpb200_200Ltnt_512_20240626_LAST_ep_200.pt
