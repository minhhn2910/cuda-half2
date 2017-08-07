# Half datatype conversion tool for CUDA programs
### Overview
This tool will help you to convert your program from the version using *float* to *half* and *half2*.
It is written in Clang libtooling (version 4.0) because that is the only option I can find to parse CUDA code easily for now.
All contribution and pull requests are welcome.

### Requirements 
- Clang 4.0, I have not tested it on other versions of Clang but I will make it compatible to newer versions if necessary.
- G++ (g++ build is optional, you can build the tool with clang using the bash script files provided
- The tutorial on how to install Clang and setup your system can be found here [https://clang.llvm.org/docs/LibASTMatchersTutorial.html]
- Linux & Unix OS for the bash scripts to run, otherwise you have to write your own Windows `.bat` scripts  to invoke clang plugins with appropriate flags.
### Building the binaries
- Changing the environment variables to reflect clang path in your system
- There are three cpp files to be built [detect_half2_vars.cpp, half2_rewrite.cpp, half_rewrite.cpp] , you can build them using clang or g++
- Example using clang to build detect_half2_vars.cpp: `./build_clang.sh detect_half2_vars`
- or you can do the same process using g++ (optional): `./build_g++.sh detect_half2_vars`

### Running the example
- First step is to identify the floating point variables that can be converted to half precision: 
`./detect_half2_vars test/vectorAdd.cu`
- After completing this step, you should have `half2VarsList.txt` file generated with the content is the list of function and floating point variables in them
- Next step is to run the conversion tool, if you want to use *half* data type with limited performance, use the file compiled from `half_rewrite.cpp`.
to generate the version using *half2* datatype: `./run_rewrite.sh test/vectorAdd.cu`

### Todo
Update sh files with environment variables











