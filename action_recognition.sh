#!/usr/bin/env bash

# Changing Directory
#sleep 30
if [[ $USER == "synergy" ]]; then
    sudo $0
fi
cd /home/synergy/audio-pose-action-recognition

# Activate conda env
source /home/synergy/.bashrc
echo "Activate conda environment"
source /home/synergy/anaconda3/bin/activate vax
python --version

sleep 2
# Start audio activity recognition
echo "Start Audio Activity Recognition"
chmod +x run_audio_action_recognition.py
python run_audio_action_recognition.py &

echo "Sleeping for 5 secs"
sleep 5

# Start video activity recognition
echo "Start Video Activity Recognition"
chmod +x remote_pose_action_recognition.py
python remote_pose_action_recognition.py &
sleep 5

wait
