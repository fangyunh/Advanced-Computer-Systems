

# Dictionary Encoding

main.cc: The main logic of how we run the test script

encoder.hh and .cc: Encoder we used for dictionary encoding

query.hh and .cc: The Query mechanism with and without SIMD

vanilla_search.hh and .cc: Vanilla baseline we used to compare with Query

The raw data "Column.txt" is not included in the folder. Install it to run the program correctly.

## How to Run the Test

1. Makefile: use command "**make**" to generate executable file. "make clean" to clean compiled file. Run commands below: {} includes all options for query (SIMD) and vanilla search. -v for vanilla method, -q for query (SIMD), and no options for query (without SIMD)

```cmd
./run -t [threads num] {-v | -q} 
```



2. To edit the search **exact query** and **prefix** you want, you need to edit **line 111, 112** for Vanilla baseline and **line 170, 171** for Query in main.cc. Clean and recompile it to rerun.
