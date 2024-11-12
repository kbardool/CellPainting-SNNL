python AE_batch.py 	\
            --runmode       baseline \
            --config        ./hyperparameters/ae_cp_150_512_cpb.yaml \
            --epochs        800 \
            --prim_opt      \
            --lr            1.00e-04 \
            --cpb           200 \
            --seed          4321 \
            --gpu_id        1 \
            --wandb         \
            # --run_id   	zmm6o058 \
            # --ckpt     	AE_baseline_20240617_scpb200-150Ltnt_512__ep_450.pt
