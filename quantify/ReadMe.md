# Quantify
This part is not of the system, but two useful tools to test the system performance except key metrics (accuracy, latency) we care 

## Memory
memory.py run for a while and log the memory usage through psutils during the period

## Energy
We can measure such energies.
* main energy
* CPU energy mainly used for decision engine and compression
* DDR energy mainly used for memory access
* GPU energy mainly used for DNN computation on edge
* SoC energy kept for OS
* Wifi energy used for transferring