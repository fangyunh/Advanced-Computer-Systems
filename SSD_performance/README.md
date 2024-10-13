# SSD Performance Test

OS: Windows 11

The program will automatically generates the FIO commands under different configuration. 



### Preparation

Create a .dat (1GB) as partition for test:

```powershell
 fsutil file createnew .\fio_testfile.dat 1073741824
```

Put fio.exe under folder ./fio:

Path of fio.exe: ./fio/fio.exe



### Codes

**fio_main.cc**: The main logic of the program. Users can add or modify configurations for generated FIO commands.

**fio_helper.cc**: Implementation of how the commands generate.

**fio_helper.hh**: Headers of helper functions. 
