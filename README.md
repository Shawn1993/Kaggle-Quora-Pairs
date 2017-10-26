# Kaggle Competition - Quora Pairs

## Introduction
具体请参考文章[《分分钟带你杀入Kaggle Top 1%》](https://zhuanlan.zhihu.com/p/27424282?group_id=906379995606454272)

## File Struture

Every directory starts with "Stage" has three sub-directory, Code, Input and Output .

	Stage0: data preprocessing  
		- Input: Contain two files, "train.csv" renamed by raw train file, "test.csv" renamed by raw test file.
		- Code: data_process.py for different process method.
		- Output: 
	Stage1:
		- Input: Output from Stage0
		- Code: handcraft feature and deep learning feature extraction
		- Output:
	Stage2:
		- Input: Output from Stage1
		- Code: unlinear ensemble, such as LightGBM, RandomForest.
		- Output:
	Stage3:
		- Input: Output from Stage2
		- Code: Ensemble Selection
		- Output: final result.


## Note 
The code is dirty and leaky. Just for reading.