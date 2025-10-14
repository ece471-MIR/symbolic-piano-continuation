#!/bin/bash
#
#SBATCH --job-name=sym-music-cont_train
#SBATCH --output=/zooper2/$USER/mir/sym_music_cont/output.txt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1-1
#SBATCH --mem=64gb

echo "docker_train.sh start ---"
docker build /zooper2/$USER/mir/sym_music_cont \
 -t sym_music_cont_train
docker run -it --rm \
 -v /zooper2/$USER/mir/sym_music_cont/sym-music-gen/runs:/sym-music-gen/runs \
 -v /zooper2/$USER/mir/sym_music_cont/sym-music-gen/data/filtered:/sym-music-gen/data/filtered \
 sym_music_cont_train
echo "docker_train.sh completed ---"
