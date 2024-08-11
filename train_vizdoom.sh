# BASELINE 
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=extrinsic --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=extrinsic --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=extrinsic --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=icm --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=icm --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=icm --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=e3b --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=e3b --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=e3b --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=rnd --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=rnd --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=rnd --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=pseudocounts --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=pseudocounts --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=pseudocounts --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=ngu --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=ngu --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=ngu --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=re3 --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=re3 --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=re3 --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=ride --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=ride --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=ride --seed=3

sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=disagreement --seed=1
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=disagreement --seed=2
sbatch train_long --env_id=MyWayHome-v1 --intrinsic_reward=disagreement --seed=3

# Q1 OBS_RMS 
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=icm --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=icm --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=icm --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=e3b --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=e3b --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=e3b --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=rnd --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=rnd --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=rnd --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=pseudocounts --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=pseudocounts --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=pseudocounts --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=ngu --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=ngu --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=ngu --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=re3 --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=re3 --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=re3 --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=ride --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=ride --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=ride --seed=3

sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=disagreement --seed=1
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=disagreement --seed=2
sbatch train_long --obs_rms --env_id=MyWayHome-v1 --intrinsic_reward=disagreement --seed=3