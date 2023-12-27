# Docker
In order to facilitate running our PyTorch/CUDA code, we are making available a Docker image (which can be used to launch containers). If you are unfamiliar with Docker, you can think of it as a lightweight terminal-only virtual machine with hardware passthrough and no persistent hard drive. Instead, files are accessed by mounting the host machine project's `src` and `rsc` directories in the Docker container's home directory.

For more information about Docker, please see the [Docker 101 Tutorial](https://www.docker.com/101-tutorial/).

- To create the Docker image, run the `docker_build.ps1` script
- To run the container, use `docker_run.ps1`
- The admin username is `user` and the password is `pass`. Details can be changed in the `Dockerfile`.

All PowerShell scripts contain just Docker commands which can be trivially converted into shell scripts.
