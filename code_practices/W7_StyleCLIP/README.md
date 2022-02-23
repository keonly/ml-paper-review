# styleCLIP-pytorch 

*Only includes global manipulation among three methods introduced in StyleCLIP*
--------------------------------------------------------------------------------
This code mostly relies on pytorch implementation of 
1. Style Space Analysis https://github.com/xrenaa/StyleSpace-pytorch.git 
2. styleCLIP official implementation [Global Manipulation] https://github.com/orpatashnik/StyleCLIP.git
 
### Set docker environment
  ```
  bash docker.sh
  ```
### Command line input
 ```
 python styleCLIP.py --neutral " A woman with hair" --target "A woman with red hair" --alpha 3.5 --beta 0.1
 ```
 *Alpha: Strength of manipulation - ranges from -10 to 10 by interval 0.1*
 
 *Beta: Manipulation direction - ranges from 0.08 to 3 by interval 0.01* 
 
 
