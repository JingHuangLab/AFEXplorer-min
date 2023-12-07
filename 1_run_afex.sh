# ./data/b7ie18 is the directory where the features.pkl file locates, as well as the predicted outcome.

python ./afexplore/afexplore_optim.py \
                            --rawfeat_dir ./data/b7ie18 \
                            --output_dir ./data/b7ie18 \
                            --afparam_dir ./afexplore/data \
                            --nsteps 10 \
                            --nclust 128 # 512