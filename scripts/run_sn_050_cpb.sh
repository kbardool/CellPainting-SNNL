python AE_batch.py 	\
            --runmode       snnl \
            --config        ./hyperparameters/ae_sn_050_512_cpb.yaml \
		    --epochs        660 \
            --lr            1.00e-04 \
            --cpb           200 \
            --seed          4321 \
            --wandb         \
            --gpu_id        1 \
            # --run_id   	ff4iokns\
            # --ckpt     	AE_snnl_scpb200-050Ltnt_512_20240624_LAST_ep_040.pt
