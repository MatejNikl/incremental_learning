# incremental_learning
My work at [GoodAI](http://www.goodai.com/) during the [Datalab Summercamp](http://datalab.fit.cvut.cz/)

# Example usage:
### 0. cd to the source directory
```bash
cd .../incremental_learning/task
```
### 1. Create convenient soft links
```bash
ln -s .../TrainingData/D1 train_data
ln -s .../TestingData/D1 test_data
```
### 2. Convert images from *.bmp to *.png
note: requires morgify + deletes original bmp files without checking, be careful
```bash
./to_png.sh train_data/* test_data/*
```
### 3. Create datasets
creates z-normalized train_data/SCT?.t7 as well as test_data/SCT?.t7
```bash
./create_datasets.sh train_data/SCT?
```
### 4. Create shared and specific parts for those datasets
```bash
mkdir net
th nncreator.lua --layers 120,120 train_data/SCT1.t7 --dir net
th nncreator.lua --use-shared net/shared.t7 train_data/SCT?.t7 --dir net
```
### 5. Train!
```bash
./run.sh experiment
```
### 6. Test!
```bash
./collect_percentages.sh experiment
```

# Required packages:
```bash
luarocks install csvigo
luarocks install image
luarocks install lua-glob-pattern
luarocks install luafilesystem
luarocks install luaposix
luarocks install nn
luarocks install nngraph
luarocks install optim
luarocks install torchnet
```
