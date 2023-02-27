# What is Docker?

First of all, what is Docker? Docker is a platform for building, shipping, and running applications in containers. Containers are a way to package software in a lightweight, portable way that makes it easy to run on different systems. Docker provides an easy way to manage and deploy containerized applications, and can be used on a wide range of platforms and operating systems.



# Docker CLI Cheat Sheet

![alt text](Images/docker-cli-cheat-sheet.png)


## IMAGES

Docker images are a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings

#### 1 - Build an Image from a Dockerfile

```
$ docker build –t  <image_name>
```

#### 2 - List local images

```
$ docker images
```

#### 3 - Delete an image

```
$ docker rmi <image_name>
```

#### 4 - Remove all unused images

```
$ docker image prune
```

## DOCKER HUB
Docker Hub is a service provided by Docker for finding and sharing container images with your team. Learn more and find images at https://hub.docker.com 

#### 1 - Login into Docker

```
$ docker login –u  <username>
```

#### 2 - Publish an image to Docker Hub

```
$ docker push <username>/<image_name>
```

#### 3 - Search Hub for an image

```
$ docker search <image_name>
```
#### 4 - Pull an image from a Docker Hub

```
$ docker pull <image_name>
```


## CONTAINERS
A container is a runtime instance of a docker image. A container will always run the same, regardless of the infrastructure. Containers isolate software from its environment and ensure that it works uniformly despite differences for instance between development and staging.


#### 1 - Create and run a container from an image, with a custom name: 


```
$ docker run --name  <container_name>  <image_name>
```

#### TIP: WHAT RUN DID

- Looked for image called '<'image_name'>' in image cache
- If not found in cache, it looks to the default image repo on Dockerhub
- Pulled it down (latest version), stored in the image cache
- Started it in a new container



#### 2 - Run a container with and publish a container’s port(s) to the host:


```
$ docker run –p  <host_port>:<container_port>   <image_name>
```

#### 3 - Run a container with a mounted volume:


```
$ docker run –it –v  <host_path>:<container_path>   <image_name>
```
#### 4 - Run a container in the background 


```
$ docker run –d <image_name>


```
#### 5 - Start or stop an existing container: 


```
$ docker start|stop <container_name>   ( or  <container-id> )
```

#### 6 - Remove a stopped container:


```
$ docker rm  <container_name>  
```

#### 7 - Open a shell inside a running container: 


```
$ docker exec –it  <container_name>  sh
```

#### 8 - Fetch and follow the logs of a container:


```
$ docker logs –f <container_name>
```

#### 9 - To list currently running containers: 


```
$ docker ps
```

#### 10 - List all docker containers (running and stopped): 

```
$ docker ps –all
```

#### 11 - View resource usage stats:

```
$ docker container stats
```


