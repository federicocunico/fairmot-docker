# docker run --privileged --gpus '"device=1"' fairmot-cuda11 /root/miniconda3/envs/fairmot/bin/python detect.py
# docker run --privileged --gpus all fairmot-cuda11 /root/miniconda3/envs/fairmot/bin/python detect.py
# docker run --privileged --gpus all -v `pwd`:/video_out fairmot-cuda11 /root/miniconda3/envs/fairmot/bin/python detect.py
# docker run --privileged --gpus all -it -v `pwd`:/video_out fairmot-cuda11
# docker run --security-opt seccomp=unconfined --gpus all -it -v `pwd`:/video_out fairmot-cuda11
docker run --privileged --gpus all -it -v `pwd`:/video_out fairmot-cuda11
