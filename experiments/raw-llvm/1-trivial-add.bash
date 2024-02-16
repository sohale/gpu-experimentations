
llc -filetype=obj 1-trivial-add.llvm -o "1-trivial-add.o"

clang "1-trivial-add.o" -o addExecutable

