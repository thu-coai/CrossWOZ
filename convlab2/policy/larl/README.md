# README

### Folders

· data/: data for training

· experiments_woz/: training script for RL and SL

· latent dialog/:LaRL model implementation



For model training, go to experiments_woz directory and follow the steps below:

### Step 1: Supervised Learning

    - sl_word: train a standard encoder decoder model using supervised learning (SL)
    - sl_cat: train a latent action model with categorical latetn varaibles using SL.
    - sl_gauss: train a latent action model with gaussian latent varaibles using SL.

### Step 2: Reinforcement Learning
Set the system model folder path in the script:
       
    folder = '2019-04-15-12-43-05-sl_cat'
    epoch_id = '8'

And then set the user model folder path in the script
    
    sim_epoch_id = '5'
    simulator_folder = '2019-04-15-12-43-38-sl_word'  # set to the log folder of the user model

Each script is used for:

    - reinforce_word: fine tune a pretrained model with word-level policy gradient (PG)
    - reinforce_cat: fine tune a pretrained categorical latent action model with latent-level PG.
    - reinforce_gauss: fine tune a pretrained gaussian latent action model with latent-level PG.