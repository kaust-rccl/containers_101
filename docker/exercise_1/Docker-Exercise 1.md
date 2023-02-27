# Exercise 1

![alt text](Images/image.png)

# Solution

![alt text](Images/image-2.png)


## Continuumio/miniconda3 docker image

#### 1 - Download the continuumio/miniconda3 image

```
$ docker pull continuumio/miniconda3

```

#### 2 - Create and Run a container named my-first-python-script & get into the bash terminal

```
$ docker run -it --name my-first-python-script continuumio/miniconda3
```

#### 3 - Create and execute a python script inside the ubuntu container "my-first-python-script"

```
$ apt update
$ apt install nano
$ nano hello.py
$ print('Helllo World')
```
```
$ python hello.py
$ exit
```

#### 4 - Execute the python script from outside using the container "my-first-python-script"

##### - List all docker containers (running and stopped)


```
$ docker ps -a
```

##### - Start the existing container "my-first-python-script" & check if it's running

```
$ docker start my-first-python-script
$ docker ps
```

##### - Execute the python script using the container "my-first-python-script"

```
$ docker exec my-first-python-script python hello.py
```

#### 5 - Remove the container named "my-first-python-script"

```
$ docker rm my-first-python-script
```

#### 6 - Delete the image continuumio/miniconda3 

```
$ docker rmi continuumio/miniconda3
```