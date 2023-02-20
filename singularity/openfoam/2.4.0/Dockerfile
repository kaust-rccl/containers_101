FROM ubuntu:16.04
# General system updates
RUN apt-get update && apt-get install -y \
    git build-essential gfortran curl wget vim

# Install MPICH 3.1.4 and cleanup  
WORKDIR /tmp
RUN curl  http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz --output mpich-3.1.4.tar.gz 
RUN tar xvf mpich-3.1.4.tar.gz
	
WORKDIR /tmp/mpich-3.1.4
RUN ./configure MPILIBNAME=mpich --prefix=/usr/local CFLAGS="-fPIC" CXXFLAGS="-fPIC" FFLAGS="-fPIC"
RUN make -j 8 VERBOSE=1
RUN make install
WORKDIR /
RUN rm -rf /tmp/mpich*
RUN ldconfig 
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# Update OpenFOAM specific system libraries
RUN apt-get install -y \
	flex bison cmake zlib1g-dev libboost-system-dev libboost-thread-dev libcgal-dev \
    gnuplot libreadline-dev libncurses-dev libxt-dev \
    qt4-dev-tools libqt4-dev libqt4-opengl-dev freeglut3-dev libqtwebkit-dev libglu1-mesa-dev \
    python python-dev 
    
    
RUN mkdir /app
ARG OFINSTDIR=/app/OpenFOAM
ARG OFVERSION=2.4.x
WORKDIR $OFINSTDIR

# get source
RUN git clone git://github.com/OpenFOAM/OpenFOAM-2.4.x.git \
	&& git clone https://github.com/OpenFOAM/ThirdParty-2.4.x.git

# definitions for the pref.sh and bashrc files. perf.sh, if exists, is sourced by bashrc

ARG OFPREFS=${OFINSTDIR}/OpenFOAM-${OFVERSION}/etc/prefs.sh
ARG OFBASHRC=${OFINSTDIR}/OpenFOAM-${OFVERSION}/etc/bashrc

RUN head -23 ${OFINSTDIR}/OpenFOAM-${OFVERSION}/etc/config/example/prefs.sh > $OFPREFS \
	&& echo '#---------------------------#' >> ${OFPREFS} \
	&& echo 'export WM_MPLIB=SYSTEMMPI' >> ${OFPREFS} \
	&& echo 'export MPI_ROOT="/usr/local"' >> ${OFPREFS} \
	&& echo 'export MPI_ARCH_FLAGS="-DMPICH_SKIP_MPICXX"' >> ${OFPREFS} \
	&& echo 'export MPI_ARCH_INC="-I${MPI_ROOT}/include"' >> ${OFPREFS} \
	&& echo 'export MPI_ARCH_LIBS="-L${MPI_ROOT}/lib${WM_COMPILER_LIB_ARCH} -L${MPI_ROOT}/lib -lmpich -lrt"' >> ${OFPREFS} 
 
 # modify bashrc
 RUN 	sed -i '/^foamInstall=$HOME.*/afoamInstall='"${OFINSTDIR}" ${OFBASHRC} \
 	 && sed -i '0,/^foamInstall=$HOME/s//# foamInstall=$HOME/' ${OFBASHRC} \
	# && sed -i 's;export WM_PROJECT_USER_DIR=\$HOME/\$WM_PROJECT/\$USER-\$WM_PROJECT_VERSION;export WM_PROJECT_USER_DIR=/workdir/\$WM_PROJECT/ofuser-\$WM_PROJECT_VERSION;g' ${OFBASHRC} \ 
 	 && echo ''	
# Download paraview (taken from wiki instructions)
 RUN . ${OFBASHRC} \
	&& cd $WM_THIRD_PARTY_DIR \
	&& mkdir download \
	&& wget --no-check-certificate -P download http://www.paraview.org/files/v4.1/ParaView-v4.1.0-source.tar.gz \
	&& tar -xzf download/ParaView-v4.1.0-source.tar.gz \
	&& mv ParaView-v4.1.0 ParaView-4.1.0 \
	&& rm -rf download
 	
# Paraview compilation (Adapted alternative instructions from OpenFoamWiki)
 RUN . ${OFBASHRC} \
	 && cd $WM_THIRD_PARTY_DIR \
	 && export QT_SELECT=qt4 \
	 && sed -i -e 's=//#define GLX_GLXEXT_LEGACY=#define GLX_GLXEXT_LEGACY=' ParaView-4.1.0/VTK/Rendering/OpenGL/vtkXOpenGLRenderWindow.cxx \
	 && cd ParaView-4.1.0 \
	 && wget --no-check-certificate http://www.paraview.org/pipermail/paraview/attachments/20140210/464496cc/attachment.bin -O Fix.patch \
	 && patch -p1 < Fix.patch \
	 && cd VTK \
	 && wget --no-check-certificate https://github.com/gladk/VTK/commit/ef22d3d69421581b33bc0cd94b647da73b61ba96.patch -O Fix2.patch \
	 && patch -p1 < Fix2.patch \
	 && cd $WM_THIRD_PARTY_DIR \
	 && ./makeParaView4 -python -mpi -python-lib /usr/lib/x86_64-linux-gnu/libpython2.7.so.1.0 2>&1 | tee log.makePV
	 
# Install Third Party tools
 RUN . ${OFBASHRC} \
	 && cd $WM_THIRD_PARTY_DIR \
	 && mkdir download \
	 && wget --no-check-certificate -P download https://gforge.inria.fr/frs/download.php/file/34099/scotch_6.0.3.tar.gz \
	 && tar -xzf download/scotch_6.0.3.tar.gz \
	 && rm -rf download

 ARG OFCGAL=${OFINSTDIR}/OpenFOAM-${OFVERSION}/etc/config/CGAL.sh
 RUN . ${OFBASHRC} \
	 && cp ${OFCGAL} ${OFCGAL}.original \
	 && sed -i '/^cgal_version=.*/acgal_version=cgal-system' ${OFCGAL} \
	 && sed -i '0,/^cgal_version/s//# cgal_version/' ${OFCGAL} \ 
	 && sed -i 's,export BOOST_ARCH_PATH,# export BOOST_ARCH_PATH,' ${OFCGAL} \
	 && sed -i 's,export CGAL_ARCH_PATH,# export CGAL_ARCH_PATH,' ${OFCGAL} \
	 && echo ''

 RUN . ${OFBASHRC} \
	&& cd $WM_THIRD_PARTY_DIR \
	&& ./Allwmake 2>&1 | tee log.Allwmake
	
# Fix for the flex version check 
 RUN . ${OFBASHRC} \
	&& cd $WM_PROJECT_DIR \
	&& find src applications -name "*.L" -type f -exec cp \{} \{}.original \; \
	&& find src applications -name "*.L" -type f | xargs sed -i -e \
	    's=\(YY\_FLEX\_SUBMINOR\_VERSION\)=YY_FLEX_MINOR_VERSION < 6 \&\& \1='

# OpenFOAM compilation 
 ENV WM_NCOMPPROCS=4
 RUN . ${OFBASHRC} \
	&& cd $WM_PROJECT_DIR \
	&& export QT_SELECT=qt4 \
	&& ./Allwmake 2>&1 | tee log.Allwmake

# Obtaining a summary
 RUN . ${OFBASHRC} \
	&& cd $WM_PROJECT_DIR \
	&& export QT_SELECT=qt4 \
	&& ./Allwmake 2>&1 | tee log.AllwmakeSummary
WORKDIR /workdir

RUN sed -i 's;export WM_PROJECT_USER_DIR=\$HOME/\$WM_PROJECT/\$USER-\$WM_PROJECT_VERSION;export WM_PROJECT_USER_DIR=/workdir/\$WM_PROJECT/ofuser-\$WM_PROJECT_VERSION;g' ${OFBASHRC} 

ENV WM_LINK_LANGUAGE=c++ \
FOAM_SOLVERS="/app/OpenFOAM/OpenFOAM-2.4.x/applications/solvers" \
FOAM_EXT_LIBBIN="/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64GccDPOpt/lib" \
FOAM_APPBIN=/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/bin \
FOAM_TUTORIALS=/app/OpenFOAM/OpenFOAM-2.4.x/tutorials \
FOAM_JOB_DIR=/app/OpenFOAM/jobControl \
WM_ARCH=linux64 \
WM_OSTYPE=POSIX \
MPI_ARCH_PATH=/usr/local \
WM_THIRD_PARTY_DIR=/app/OpenFOAM/ThirdParty-2.4.x \
FOAM_SITE_APPBIN=/app/OpenFOAM/site/2.4.x/platforms/linux64GccDPOpt/bin \
FOAM_APP=/app/OpenFOAM/OpenFOAM-2.4.x/applications \
WM_CXXFLAGS="-m64 -fPIC" \
WM_CFLAGS="-m64 -fPIC" \
WM_PROJECT_VERSION=2.4.x \
FOAM_SITE_LIBBIN=/app/OpenFOAM/site/2.4.x/platforms/linux64GccDPOpt/lib \
LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64Gcc/gperftools-svn/lib:/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64Gcc/ParaView-4.1.0/lib/paraview-4.1:/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/lib/mpi-system:/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64GccDPOpt/lib/mpi-system:/workdir/OpenFOAM/ofuser-2.4.x/platforms/linux64GccDPOpt/lib:/app/OpenFOAM/site/2.4.x/platforms/linux64GccDPOpt/lib:/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/lib:/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64GccDPOpt/lib:/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/lib/dummy:/usr/local/lib" \
MPI_BUFFER_SIZE=20000000 \
WM_COMPILER_LIB_ARCH=64 \
FOAM_SRC="/app/OpenFOAM/OpenFOAM-2.4.x/src" \
FOAM_SIGFPE= \
ParaView_DIR="/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64Gcc/ParaView-4.1.0" \
WM_PROJECT_INST_DIR=/app/OpenFOAM \
FOAM_UTILITIES="/app/OpenFOAM/OpenFOAM-2.4.x/applications/utilities" \
WM_CXX=g++ \
WM_PROJECT_DIR="/app/OpenFOAM/OpenFOAM-2.4.x" \
WM_NCOMPPROCS=4 \
WM_PROJECT=OpenFOAM \
WM_LDFLAGS=-m64 \
WM_COMPILER=Gcc \
WM_MPLIB=SYSTEMMPI \
PATH="\${PATH}:/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64Gcc/gperftools-svn/bin:/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64Gcc/ParaView-4.1.0/bin:/workdir/OpenFOAM/ofuser-2.4.x/platforms/linux64GccDPOpt/bin:/app/OpenFOAM/site/2.4.x/platforms/linux64GccDPOpt/bin:/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/bin:/app/OpenFOAM/OpenFOAM-2.4.x/bin:/app/OpenFOAM/OpenFOAM-2.4.x/wmake:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
FOAM_USER_LIBBIN="/workdir/OpenFOAM/ofuser-2.4.x/platforms/linux64GccDPOpt/lib" \
MPI_ARCH_LIBS="-L/usr/local/lib -L/usr/local/lib -lmpich -lrt" \
WM_CC=gcc \
FOAM_ETC="/app/OpenFOAM/OpenFOAM-2.4.x/etc" \
PV_PLUGIN_PATH="/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/lib/paraview-4.1" \
FOAM_INST_DIR="/app/OpenFOAM" \
ParaView_INCLUDE_DIR="/app/OpenFOAM/ThirdParty-2.4.x/platforms/linux64Gcc/ParaView-4.1.0/include/paraview-4.1" \
FOAM_MPI=mpi-system \
FOAM_LIBBIN="/app/OpenFOAM/OpenFOAM-2.4.x/platforms/linux64GccDPOpt/lib" \
WM_COMPILE_OPTION=Opt \
ParaView_VERSION=4.1.0 \
WM_DIR=/app/OpenFOAM/OpenFOAM-2.4.x/wmake \
ParaView_MAJOR=4.1 \
FOAM_SETTINGS= \
MPI_ARCH_INC=-I/usr/local/include \
MPI_ARCH_FLAGS=-DMPICH_SKIP_MPICXX \
MPI_ROOT=/usr/local \
WM_PROJECT_USER_DIR=/workdir/OpenFOAM/ofuser-2.4.x \
FOAM_RUN=/workdir/OpenFOAM/ofuser-2.4.x/run \
WM_OPTIONS=linux64GccDPOpt \
WM_PRECISION_OPTION=DP \
WM_ARCH_OPTION=64 \
FOAM_USER_APPBIN="/workdir/OpenFOAM/ofuser-2.4.x/platforms/linux64GccDPOpt/bin"
