	python AE_batch.py 	 --config ./hyperparameters/ae_sn_250_512_cpb.yaml \
            --runmode       snnl \
            --epochs        200 \
            --lr            1.00e-04 \
            --temp_opt      --prim_opt \
            --temp_lr       1.00e-05 \
            --temp          0.005 \
	        --cpb       200 \
            --seed          4321 \
            --wandb         \
            --gpu_id        0 \
            # --run_id      qw52e4d2 \
            # --ckpt        AE_snnl_dcpb200_200Ltnt_512_20240626_LAST_ep_200.pt
