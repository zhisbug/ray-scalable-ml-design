# Spacy-Ray benchmark experiments

## Setup:
The experiment uses the project from the original spacy-ray repo. The benchmark was run on CMU's cluster, with one GPU on each node. 

## How to run:
When running for the first time, run ```startup.sh```. Then, configure the machine address, number of workers and GPU's in ```run.sh``` and run the script.

## Benchmark Results

### On Hao's PC:

Batch Size: 1000, 10 sentences per document, evaluate every 200 iteration

| #GPU     | Total Throughput (words/sec)   | Average Throughput (words/sec) |Total Throughput (sentences/sec) | Average Throughput (sentences/sec) | Scalability |
| :------------- | :----------: | :----------:|:----------:|:----------:|:----------:| -----------: |
|  1 | 4439.01   |  4408.82 |18.30  | 18.18 | N/A| 
| 2   | 3657.02 | 3603.41  |15.08 | 14.86 | 0.82

### On Cluster:

Batch size 250, 1 sentence per document, evaluate every 1000 iteration, run 3 times and calculate average and sample std

| #GPU     | Throughput (words/sec) | Standard Deviation (words/sec) | Throughput (sentences/sec) | Standard Deviation (sentences/sec)^2 | Convergence |
| :------------- | :----------: | :----------:|:----------: | :----------:|:----------:| :-----------: |
| 1   | 1451.02 | 31.43 | 59.85 | 1.30 | 0.54 |
| 2   | 985.89 | 71.85 | 40.67 | 2.96 | 0.62 |
|4| 1452.82 |  71.09 | 59.93 | 2.93 | 0.57|
|8 | 2632.88 |100.87 | 108.60|4.16 | 0.52 |
|12 | 3888.65 |163.23 | 160.40|6.73 | 0.51 | 
|16 | 5322.44 |143.21 | 219.54| 5.91 | 0.49 | 

Batch size 1000, 1 sentence per document, evaluate every 1000 iteration, run 3 times and calculate average and sample std

| #GPU     | Throughput (words/sec) | Standard Deviation (words/sec) | Throughput (sentences/sec) | Standard Deviation (sentences/sec)^2 | Convergence |
| :------------- | :----------: | :----------:|:----------: | :----------:|:----------:| :-----------: |
| 1   | 4275.57 | 77.77 | 175.77 | 3.20 | 0.71 |
| 2   | 3226.64 | 198.43 | 132.65 | 8.16 | 0.74 |
|4| 3872.50 |  	64.98 | 159.2 | 	2.68 | 0.69|
|8 | 7772.29 |94.68 | 319.52 | 3.89 | 0.65 |
|12 | 11815.41 |229.57 | 485.73 |9.44 | 0.63 | 
|16 | 15717.95 |328.02 | 646.16 | 13.49 | 0.62 | 