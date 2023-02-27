# Containers 101 - Docker 

Welcome to the GitHub repository! This repository contains different materials for the introduction to containers session.

There are examples here for : 
- Docker commands 
- Dockerfiles 

This examples are to run on your local machine. With docker installed.

We hope you find these resources helpful in your Docker journey.

## Example 1 : Docker commands

Here, you'll find resources to help you get started with Docker, including:

- A brief introduction to Docker and how to manipulate images and containers ([Docker CLI cheat sheet](./Docker-CLI-Cheat-sheet.md))
- [A Docker - Demo](./Docker-Demo.md)
- [Docker - Exercise 1](./Docker-Exercise-1.md)


## Example 2 : Dockerfiles
If you are running this example on a local machine with apple silicon, you need to run the folling command to build the image. 

```docker build -t customjupyter --platform linux/amd64 --no-cache .```

To run the container, you need to run the following command. 

```docker run -it -p 8888:8888 customjupyter```

## Authors

- [Dr. Didier Barradas Bautista](https://www.github.com/octokatherine)
- [Dr. Abdelghafour Halimi](https://www.ahalimi.com/)
- [Dr. Kadir Akbudak ](https://www.hpc.kaust.edu.sa/team)
- [Dr. Mohsin Ahmed Shaikh](https://www.hpc.kaust.edu.sa/team)


## Documentation

[Ibex training](https://www.hpc.kaust.edu.sa/ibex/training
)

[Shaheen training](https://www.hpc.kaust.edu.sa/training
)

[KSL How-To repository](https://kaust-supercomputing-lab.atlassian.net/l/cp/tAG1wkA0)




## Support

For support, email ibex@hpc.kaust.edu.sa , help@hpc.kaust.edu.sa or join [Ibex slack channel](kaust-ibex.slack.com 
)



## 🔗 Links

[KAUST Core Labs](https://corelabs.kaust.edu.sa/
) : 
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/kaust-core-labs/about/) [![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/kaust_corelabs)

[KAUST Supercomputing Lab](https://www.hpc.kaust.edu.sa/) : 
[![KAUST_HPC](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/KAUST_HPC) 

[KAUST Vizualization Core Lab](https://corelabs.kaust.edu.sa/labs/detail/visualization-core-lab) :
[![KVL](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/KAUST_Vislab)  
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UCR1RFwgvADo5CutK0LnZRrw?style=social)](https://www.youtube.com/channel/UCR1RFwgvADo5CutK0LnZRrw)
