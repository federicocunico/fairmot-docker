conda create --name fairmot python=3.7
conda activate fairmot
git clone https://github.com/microsoft/FairMOT

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
cd FairMOT
pip install cython
pip install -r requirements.txt

git clone https://github.com/ifzhang/DCNv2.git
cd DCNv2 && git checkout ab4d98efc0aafeb27bb05b803820385401d9921b && ./make.sh

cd ..
mkdir models
cd models
# download
echo download https://drive.google.com/open?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu
# wget "https://drive.google.com/u/0/uc?export=download&confirm=Y4qc&id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu"
