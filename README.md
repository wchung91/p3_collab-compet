#Project Name: Collab Compet
#1. Description of Environment
The environment contains two agents playing tennis. The two agents control rackets to bounce a ball over a net. The action space of each agent is 2 dimensions which is continuous. The agent can move towards (or away from) the net and jump. Every time an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets the ball hit the ground or out of bound, it will receive a reward of -0.01. The observation space is 8 variables containing the position, velocity of the ball and the racket. Each agent receives its own local observation. The goal of each agent is to keep the ball in play. The problem is considered solved if the agents get an average score of +0.5 over 100 consecutive episodes, after taking the maximum over both agents. 


#2. Description of Installation 
We use a docker with the nvidia driver and isolate the environment. Inside the docker, we then create a virtual environment to use Python 3.6. In the virtual environment, we install pytorch and unityagents. 

#3. Installation Guide 
3.1 This installation guide assumes that 
     -OS is Ubuntu 16.04. 
     -Docker is installed  (https://docs.docker.com/engine/install/ubuntu/) 
     -Nvidia driver is installed Cuda is installed 
     -Cudnn is installed 
     -nvidia-docker is installed (https://github.com/NVIDIA/nvidia-docker) 
     -git is installed 

3.2 Clone the repository 

   git clone https://github.com/wchung91/p3_collab-compet.git

3.3 Build the dockerfile. Run the command below in the terminal and it will create an image named rl_env.

   sudo docker build . --rm -t rl_env_collab

If you use a different GPU from RTX 2080, you need to change the dockerfile. Open the dockerfile and change “pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime” to the docker image corresponding to the cuda and cudnn installed on your computer. 

3.4 Create a docker container from the docker image, but you need to change “/home/wally-server/Documents/p3_collab-compet” in the command below and then run the command. “/home/wally-server/Documents/p3_collab-compet” is the directory of the volume. You should change “/home/wally-server/Documents/p3_collab-compet” to the path to the cloned repository. That way the docker container has access to the files cloned from the repository. One you changed the command, run the command. 

   sudo docker run -it -d --gpus all -v /home/wally-server/Documents/p3_collab-compet:/workspace --name p3_collab-compet_container rl_env_cont /bin/bash

3.5 To access the container run,  
 
   sudo docker exec -it p3_collab-compet_container /bin/bash

3.6 Inside the container run the command below to initialize conda with bash 

   conda init bash

3.7 You need to close and reopen a new terminal. You can do that with the command from 3.5. Create a virtual environment named “p3_env” with python 3.6 with the following code 

   conda create -n p3_env -y python=3.6

3.8 Activate the environment “p3_env” with the command below. 

   conda activate p3_env

3.9 Inside the virtual environment, install pytorch with the command below. You’ll have to install the correct pytorch version depending on your cuda and cudnn version. 

   pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

3.10 Install unityagents with the following code. 

   pip install unityagents

3.11 Download the unity environments with the following commands. Since we are using a docker, you’ll have to use Tennis_Linux_NoVis because no display is available. 

   wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip && unzip Tennis_Linux_NoVis.zip


3.12 To run the training code, go to main.py and set “Train = 1”. Then, run the command below

   python main.py 

The code will print the average scores, and it will create a figure called “ScoresTraining.png”

3.13 To run the testing code, go to main.py and set “Train = 0”. Then, run the command below 

   python main_one.py

The code will print the average scores, and it will create a figure called “TrainedScore.png”

#4. About the Code 
main.py - contains the main method for running the code. The code is divided into training code and testing code 
maddpg.py - contains multiple agents 
ddpg.py - contains the code for a single ddpg agent, experience replay buffer, noise.
model.py - contains the actor and critic models. 
checkpoint_actor_0.pth - trained actor model for agent 0
checkpoint_critic_0.pth - trained critic model for agent 0 
checkpoint_actor_1.pth - trained actor model for agent 1
checkpoint_critic_1.pth - trained critic model for agent 1

#5. Incase code doesn't run 
Make sure in main_one.py the code below points to the right path of the "Tennis.x86_64"

   env = UnityEnvironment(file_name='./Tennis_Linux_NoVis/Tennis.x86_64')




