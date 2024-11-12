python AE_batch.py  --config./hyperparameters/ae_cp_025_512.yaml \
	                  --epochs      50 \
                      --runmode     baseline \
                      --wandb \
                      --gpu_id      2\
                      --run_id      tcvl4wdr \
                      --ckpt        AE_base_snglOpt_025Ltnt_512_20240613_ep_400.pt
