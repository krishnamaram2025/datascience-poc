# datascience-poc
This project is intended to touch and feel Data Science.

# Pre-Requisites
* Step 1: launch machine

* Step 2:login to machine
```
ssh centos@IP
```
# Without Docker
* Step 1: Install dependencies
```
sudo yum install java -y

sudo yum install python3-pip -y

sudo yum install git -y
```
* Step 2: Clone repo 
```
git clone https://github.com/krishnamaram2/datascience-poc.git && cd datascience-poc
```
* Step 3: Install python dependencies
```
sudo pip3 install -r requirements.txt
```
* Step 4: Training model
```
spark-submit training.py
```
* Step 5: Testing
```
spark-submit testing.py
```

# With Docker
* Step 1: Install Docker
```
sudo yum update -y
sudo yum install -y yum-utils device-mapper-persistent-data lvm2
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo docker --version
sudo usermod -aG docker $USER
sudo chown $USER:$USER /var/run/docker.sock
```
* Step 2: Clone repo
```
sudo yum install git -y
git clone https://github.com/krishnamaram2/datascience-poc.git && cd datascience-poc
```
* Step 3: Build docker image
```
docker build -t my-centos-spark-app .
```
* Step 4: Run docker conatiner
```
docker run -it my-centos-spark-app
```
