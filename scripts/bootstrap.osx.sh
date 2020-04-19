#
# Install python3 if it is not yet there
#
echo "Checking numpy"
hasNumpy=$(python3 -c "import numpy" 2>&1 | grep -c ModuleNotFound)
if [[ "$hasNumpy" == "1" ]];
then
    echo "Install numpy"
    python3 -m pip install --user numpy
fi

#
# Default libraries
#
brew install wget autoconf automake openssl

#
# Get root directory
#
root="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

#
# Precompile python
#
echo "Compile gpt"
python3 -m compileall ${root}/lib/gpt

#
# Create dependencies and download
#
dep=${root}/dependencies
if [ -d ${dep} ];
then
    echo "$dep already exists ; rm -rf $dep before bootstrapping again"
    exit 1
fi

mkdir -p ${dep}
cd ${dep}

#
# Lime
#
wget https://github.com/usqcd-software/c-lime/tarball/master
tar xzf master
mv usqcd-software-c-lime* lime
rm -f master
cd lime
./autogen.sh
./configure
make
cd ..

#
# Grid
#
git clone https://github.com/lehner/Grid.git
cd Grid
git checkout feature/gpt
./bootstrap.sh
mkdir build
cd build
../configure --enable-precision=double --enable-simd=AVX2 --enable-comms=none CXXFLAGS=-fPIC --with-lime=${dep}/lime
#--with-openssl=${dep}/openssl
cd Grid
make -j 4

exit

#
# cgpt
#
cd ${root}/cgpt
./make # still may need numpy headers
