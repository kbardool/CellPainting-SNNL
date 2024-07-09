python AE_batch.py 	\
            --runmode         snnl \
            --config          ./hyperparameters/ae_sn_150_512_cpb.yaml \
            --epochs          800 \
            --prim_opt        \
            --temp_opt        \
            --lr              1.00e-04 \
            --temp_lr         1.00e-05 \
            --temp            0.5 \
            --cpb             200 \
            --seed            4321 \
            --gpu_id          0 \
            --wandb           \
            # --run_id        1dneeedh \
            # --ckpt          AE_snnl_dcpb200_150Ltnt_512_20240708_1947_LAST_ep_400.pt
