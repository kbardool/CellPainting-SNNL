python AE_batch.py 	\
            --runmode       snnl \
            --config        ./hyperparameters/ae_sn_200_512_cpb.yaml \
            --epochs        400 \
            --prim_opt      \
            --lr            1.00e-04 \
            --temp_lr       1.00e-05 \
            --temp          0.5 \
            --temp_annealing \
	        --cpb           200 \
            --seed          4321 \
            --wandb         \
            --gpu_id        1 \
            --run_id        53nu5pmd \
            --ckpt          AE_snnl_dcpb200_200Ltnt_512_20240709_2022_LAST_ep_400.pt
            # --run_id        qheblbmi \
            # --ckpt          AE_snnl_dcpb200_150Ltnt_512_20240708_0612_LAST_ep_800.pt