FROM ompi:latest
RUN apt-get install -y cmake libfftw3-dev
WORKDIR /opt
RUN curl -o gromacs-2020.4.tar.gz -L http://ftp.gromacs.org/pub/gromacs/gromacs-2020.4.tar.gz && \
    tar zxvf gromacs-2020.4.tar.gz && \
    rm gromacs-2020.4.tar.gz 
WORKDIR gromacs-2020.4/build
RUN cmake -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DGMX_MPI=on -DBUILD_SHARED_LIBS=on -DGMX_FFT_LIBRARY=fftw3 -DCMAKE_BUILD_TYPE=Release .. && \
	make VERBOSE=1 && \
	make install
ENV PATH /usr/local/gromacs/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/gromacs/lib:$LD_LIBRARY_PATH 
