python AE_baseline.py 	--config  ./hyperparameters/ae_sn_150_512_cpb.yaml \
            --runmode       snnl \
            --epochs        100 \
            --prim_opt      --temp_opt      \
            --lr            1.00e-04 \
            --temp_lr       1.00e-06 \
            --temp          0.5 \
	        --cpb           200 \
            --seed          4321 \
            --wandb         \
            --gpu_id        0 \
            # --run_id        n573ysrm \
            # --ckpt          AE_snnl_dcpb200_150Ltnt_512_240628_0051_LAST_ep_200.pt
