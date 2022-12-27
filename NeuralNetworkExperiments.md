# My Learnings:

  

## 1) Getting tensorflow to work is tedious

While this looked as simple and plain as running yet another pip install command, boy, I was wrong.

 -  I started off with doing a pip install of requirements.txt. After finishing my first round of program, I hit `python traffic.py gtsrb` and I am in for a treat. **90 seconds per epoch**! By this speed, I was getting a whopping 15 minute turn-around time for a single execution - which was outrageous! 
 - I start digging in to see how can I optimize this. Initial pointers were towards `Google Collab` - but I was soon discouraged by my lack of familiarity with the interface and the necessity to upload the gtsrb folder to Gdrive to be able to use it.
 -  Other areas pointed towards a better path - Using `conda` to handle environments and packages. There were a couple of hiccups - Involving a `graphics card driver re-install` and `incompatible` `cuDNN` packages, but a few hours of digging helped me get this right and land a roughly **5-8 second per epoch** rate - which was marginally improved.
 -  I am still not sure how to get this to work in VScode terminal - I tried integrating conda environment with the Python runtime in vscode and I still get 50 seconds per epoch - I have resorted to editing the files in vscode IDE but executing them on the conda terminal. Minor inconvenience as compared to 15 minute program runs! 
  

## 2) Experimenting with the model:

I took the liberty to modify the main function to develop a better understanding of the networks and layers. I have commented out the code for my code submission but have included the artefacts - Model summaries, Accuracy and Loss plots and Model Diagrams. This was purely for improving my interpretation of the experiment.
 
## Constants

These do not change over the course of the experiment

> EPOCHS = 15 IMG_WIDTH = 30 IMG_HEIGHT = 30 NUM_CATEGORIES = 43 TEST_SIZE = 0.4

## Experiment 1: Keep it Simple, Stupid. 

 The good-old KISS Principle. Starting with a vanilla model: Conv2D(36,(3,3)) ->Maxpooling(3,3)->Flatten -> Hidden Layer(Categories*32) -> Dropout(0.5) -> Output. We get extremely high loss and low accuracy to begin with, and even after 15 iterations, loss looks pretty high. 

**Iterations**

    Epoch 1/15     2022-12-27 15:13:31.522207: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100     
    500/500 [==============================] - 7s 4ms/step - loss: 7.7904 - accuracy: 0.5135     
    Epoch 2/15     500/500 [==============================] - 2s 5ms/step - loss: 0.8368 - accuracy: 0.7796     
    Epoch 3/15     500/500 [==============================] - 2s 5ms/step - loss: 0.6078 - accuracy: 0.8447     
    Epoch 4/15     500/500 [==============================] - 2s 5ms/step - loss: 0.5156 - accuracy: 0.8712     
    Epoch 5/15     500/500 [==============================] - 2s 4ms/step - loss: 0.5015 - accuracy: 0.8801     
    Epoch 6/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4439 - accuracy: 0.8970     
    Epoch 7/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4494 - accuracy: 0.8991     
    Epoch 8/15     500/500 [==============================] - 2s 4ms/step - loss: 0.5025 - accuracy: 0.8893     
    Epoch 9/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4433 - accuracy: 0.9036     
    Epoch 10/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4829 - accuracy: 0.8996     
    Epoch 11/15     500/500 [==============================] - 2s 4ms/step - loss: 0.3898 - accuracy: 0.9204     
    Epoch 12/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4160 - accuracy: 0.9176     
    Epoch 13/15     500/500 [==============================] - 2s 4ms/step - loss: 0.3919 - accuracy: 0.9206     
    Epoch 14/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4879 - accuracy: 0.9131     
    Epoch 15/15     500/500 [==============================] - 2s 4ms/step - loss: 0.4540 - accuracy: 0.9148

**Model Summary**
|Layer (type)   |Output Shape|  Param #| 
|--|--|-- | 
|conv2d (Conv2D)   |(None, 28, 28, 36)  |1008 |  
|max_pooling2d (MaxPooling2D)| (None, 9, 9, 36)| 0|
|flatten (Flatten)| (None, 2916)| 0
|dense (Dense)| (None, 1376)| 4013792
|dropout (Dropout)| (None, 1376)| 0
|dense_1 (Dense) | (None, 43) |59211

> Total params: 4,074,011 Trainable params: 4,074,011 Non-trainable params: 0

  

**Plot:**  [Accuracy and Loss over Epochs](https://prnt.sc/Eqhx0ekRq7S0)
**Model:** [Model Diagram](https://prnt.sc/_YV19rzhREpB)

## Experiment 2: Intensify dense processing in the single hidden layer

Decided to see if I can increase hidden layer processing a substantial number of categories(Categories * 1024).  This proved to be too heavy for my system - `Out of Memory exceptions and failure in first epoch`. 

    Epoch 1/15
    2022-12-27 15:23:11.736201: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
    2022-12-27 15:23:26.722609: W tensorflow/core/common_runtime/bfc_allocator.cc:479] Allocator (GPU_0_bfc) ran out of memory trying to allocate 489.80MiB (rounded to 513589248)requested by op gradient_tape/sequential/dense/MatMul/MatMul_1
    If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation.

## Experiment 3: Taking it slightly easy on the hidden layer

Reduced the factor by 2 (Categories*512) from last experiment - This seemed to introduce a lot of processing. Number of params exploded to **65 million+** , processing took longer and accuracy and loss both suffered. Overall, a bad setup. 

**Iterations**

    Epoch 1/15  2022-12-27 15:26:09.616476: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100  
    500/500 [==============================] - 18s 26ms/step - loss: 26.9859 - accuracy: 0.4398
    Epoch 2/15  500/500 [==============================] - 13s 26ms/step - loss:1.7276 - accuracy: 0.6273  
    Epoch 3/15  500/500 [==============================] - 13s 26ms/step - loss:1.5957 - accuracy: 0.6747  
    Epoch 4/15  500/500 [==============================] - 13s 26ms/step - loss:1.3824 - accuracy: 0.7065  
    Epoch 5/15  500/500 [==============================] - 14s 29ms/step - loss:1.3534 - accuracy: 0.7013  
    Epoch 6/15  500/500 [==============================] - 15s 30ms/step - loss:  1.0155 - accuracy: 0.7694  
    Epoch 7/15  500/500 [==============================] - 15s 30ms/step - loss:  1.0470 - accuracy: 0.7832  
    Epoch 8/15  500/500 [==============================] - 14s 27ms/step - loss:  1.1468 - accuracy: 0.7628  
    Epoch 9/15  500/500 [==============================] - 13s 26ms/step - loss:  1.1106 - accuracy: 0.7752  
    Epoch 10/15  500/500 [==============================] - 13s 26ms/step - loss:  1.0825 - accuracy: 0.7984  
    Epoch 11/15  500/500 [==============================] - 13s 26ms/step - loss:  0.9876 - accuracy: 0.8034  
    Epoch 12/15  500/500 [==============================] - 13s 26ms/step - loss:  0.8303 - accuracy: 0.8278  
    Epoch 13/15  500/500 [==============================] - 13s 27ms/step - loss:  0.9999 - accuracy: 0.8149  
    Epoch 14/15  500/500 [==============================] - 16s 33ms/step - loss:  0.8452 - accuracy: 0.8350  
    Epoch 15/15  500/500 [==============================] - 14s 27ms/step - loss:  0.8257 - accuracy: 0.8532


**Model Summary**


|Layer (type) |Output Shape| Param #|
|--|--|--|
|conv2d (Conv2D)|(None, 28, 28, 36)|1008
|max_pooling2d (MaxPooling2D)|(None, 9, 9, 36)|0
|flatten (Flatten)|(None, 2916)| 0
|dense (Dense)| (None, **22016**)| **64220672**
|dropout (Dropout)| (None, 22016)| 0
|dense_1 (Dense) |(None, 43) |946731

 

>  Total params: **65,168,411** Trainable params: 65,168,411 Non-trainable params: 0

  **Plot:**  [Accuracy and Loss over Epochs](https://prnt.sc/Qo5pUuUg59if)

## Experiment 4: Recovering from Experiment 3

Changed the hidden layer processing by making densing process take (Categories * 64). This seemed to give a moderate mix of accuracy, loss and processing time.

**Iterations**

    Epoch 1/15 2022-12-27 15:39:56.225182: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    500/500 [==============================] - 8s 6ms/step - loss: 11.2384 - accuracy: 0.4909 
    Epoch 2/15 500/500 [==============================] - 3s 5ms/step - loss: 0.9384 - accuracy: 0.7815 
    Epoch 3/15 500/500 [==============================] - 3s 5ms/step - loss: 0.6236 - accuracy: 0.8569 
    Epoch 4/15 500/500 [==============================] - 3s 5ms/step - loss: 0.4982 - accuracy: 0.8903 
    Epoch 5/15 500/500 [==============================] - 3s 5ms/step - loss: 0.5030 - accuracy: 0.8903 
    Epoch 6/15 500/500 [==============================] - 3s 5ms/step - loss: 0.4936 - accuracy: 0.8977 
    Epoch 7/15 500/500 [==============================] - 3s 5ms/step - loss: 0.5420 - accuracy: 0.8953 
    Epoch 8/15 500/500 [==============================] - 3s 5ms/step - loss: 0.5383 - accuracy: 0.9007 
    Epoch 9/15 500/500 [==============================] - 3s 5ms/step - loss: 0.4686 - accuracy: 0.9086 
    Epoch 10/15 500/500 [==============================] - 3s 5ms/step - loss: 0.5908 - accuracy: 0.9046 
    Epoch 11/15 500/500 [==============================] - 3s 5ms/step - loss: 0.5242 - accuracy: 0.9112 
    Epoch 12/15 500/500 [==============================] - 3s 5ms/step - loss: 0.4249 - accuracy: 0.9279 
    Epoch 13/15 500/500 [==============================] - 3s 5ms/step - loss: 0.4979 - accuracy: 0.9195 
    Epoch 14/15 500/500 [==============================] - 3s 5ms/step - loss: 0.6709 - accuracy: 0.9102 
    Epoch 15/15 500/500 [==============================] - 3s 5ms/step - loss: 0.5034 - accuracy: 0.9225


**Model Summary**

|Layer (type)| Output Shape| Param #|
|--|--|--|
conv2d (Conv2D)| (None, 28, 28, 36) |1008
max_pooling2d (MaxPooling2D)| (None, 9, 9, 36)|0
flatten (Flatten) |(None, 2916) |0
dense (Dense) |(None, **2752**) |**8027584**
dropout (Dropout) |(None, 2752) |0
dense_1 (Dense) |(None, 43) |118379


> Total params: 8,146,971  Trainable params: 8,146,971 Non-trainable params: 0

**Plot:**[ Accuracy and Loss over Epochs](https://prnt.sc/2oH2HYJyVyXl)

## Experiment 5: Commence the Dropouts!

 Introducing dropout(0.5) between Flatten and first hidden layer. This seemed to stale the learning process. Loss remains at consistent high of 1+ and accuracy does not go above 0.8. Changing dropout value to 0.1 improved the accuracy and reduced loss though. (Refer to second plot)

**Iterations**

    
 

    Epoch 1/15 2022-12-27 15:48:06.879027: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    500/500 [==============================] - 12s 5ms/step - loss: 9.4088 - accuracy: 0.3077 
    Epoch 2/15 500/500 [==============================] - 3s 5ms/step - loss: 1.9552 - accuracy: 0.5163 
    Epoch 3/15 500/500 [==============================] - 3s 6ms/step - loss: 1.5674 - accuracy: 0.6101 
    Epoch 4/15 500/500 [==============================] - 3s 5ms/step - loss: 1.3370 - accuracy: 0.6726 
    Epoch 5/15 500/500 [==============================] - 3s 5ms/step - loss: 1.2997 - accuracy: 0.6929 
    Epoch 6/15 500/500 [==============================] - 3s 5ms/step - loss: 1.1967 - accuracy: 0.7203 
    Epoch 7/15 500/500 [==============================] - 3s 5ms/step - loss: 1.1292 - accuracy: 0.7374 
    Epoch 8/15 500/500 [==============================] - 3s 5ms/step - loss: 1.1845 - accuracy: 0.7353 
    Epoch 9/15 500/500 [==============================] - 3s 5ms/step - loss: 1.0843 - accuracy: 0.7536 
    Epoch 10/15 500/500 [==============================] - 3s 5ms/step - loss: 1.3173 - accuracy: 0.7177 
    Epoch 11/15 500/500 [==============================] - 3s 5ms/step - loss: 1.1202 - accuracy: 0.7546 
    Epoch 12/15 500/500 [==============================] - 3s 5ms/step - loss: 1.1030 - accuracy: 0.7684 
    Epoch 13/15 500/500 [==============================] - 3s 5ms/step - loss: 1.0769 - accuracy: 0.7807 
    Epoch 14/15 500/500 [==============================] - 3s 5ms/step - loss: 1.1511 - accuracy: 0.7670 
    Epoch 15/15 500/500 [==============================] - 3s 5ms/step - loss: 1.0172 - accuracy: 0.7907


**Model Summary**

Layer (type) |Output Shape |Param # 
|--|--|--|
conv2d (Conv2D) |(None, 28, 28, 36) |1008 
max_pooling2d (MaxPooling2D) |(None, 9, 9, 36) |0 
flatten (Flatten) |(None, 2916) |0 
**dropout (Dropout)** |(None, 2916)| 0|
 dense (Dense) |(None, 2752)| 8027584 |
 dropout_1 (Dropout) |(None, 2752) |0 
 dense_1 (Dense) |(None, 43) |


> Total params: 8,146,971 Trainable params: 8,146,971 Non-trainable params: 0


**Plot1(Dropout 0.5):** [Accuracy and Loss over Epochs](https://prnt.sc/Y-B1Z_ObRPYk)

**Plot2(Dropout 0.1):** [Accuracy and Loss over Epochs](https://prnt.sc/hBjr5sHh4f6q)

## Experiment 6: More hidden layer(s)!

Added a densing layer with X16 multiplier right after the first dropout in hidden layer. This improved accuracy to 0.9 and reduced loss to a degree.

**Iterations**

    Epoch 1/15 2022-12-27 15:58:35.267067: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    500/500 [==============================] - 7s 6ms/step - loss: 7.5445 - accuracy: 0.4489 
    Epoch 2/15 500/500 [==============================] - 3s 6ms/step - loss: 0.9851 - accuracy: 0.7326 
    Epoch 3/15 500/500 [==============================] - 3s 6ms/step - loss: 0.6716 - accuracy: 0.8170 
    Epoch 4/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5754 - accuracy: 0.8415 
    Epoch 5/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5451 - accuracy: 0.8570 
    Epoch 6/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4638 - accuracy: 0.8791 
    Epoch 7/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4438 - accuracy: 0.8833 
    Epoch 8/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4763 - accuracy: 0.8841 
    Epoch 9/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4244 - accuracy: 0.8889 
    Epoch 10/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4713 - accuracy: 0.8859 
    Epoch 11/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4666 - accuracy: 0.8945 
    Epoch 12/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4896 - accuracy: 0.8874 
    Epoch 13/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4105 - accuracy: 0.9045 
    Epoch 14/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5341 - accuracy: 0.8844 
    Epoch 15/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4546 - accuracy: 0.9002


**Model Summary**
Layer (type) |Output Shape |Param # 
|-|-|-|
conv2d (Conv2D) |(None, 28, 28, 36) |1008 
max_pooling2d (MaxPooling2D)| (None, 9, 9, 36)| 0 
flatten (Flatten) |(None, 2916) |0 
dropout (Dropout) |(None, 2916) |0 
dense (Dense) |(None, 2752) |8027584 
dropout_1 (Dropout) |(None, 2752) |0 
**dense_1 (Dense)** |**(None, 688)** |**1894064** 
dense_2 (Dense) |(None, 43) |29627

> Total params: **9,952,283** Trainable params: 9,952,283 Non-trainable params: 0
 
**Plot:** [Accuracy and Loss over Epochs](https://prnt.sc/xC2afwVpxNBC)

**Model:** [Model Diagram](https://prnt.sc/tnfqhkBl02T8)

## Experiment 7: YADDL

**Y**et **A**nother **D**ropout and **D**ensing **L**ayer - Added a dropout(0.5) and a densing layer(0.5) just after the dropout introduced in Experiment 6.. This did no good to loss and accuracy for 15 epochs. Both worsened by an extent.

**Iterations**


    Epoch 1/15 2022-12-27 16:06:37.054720: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    500/500 [==============================] - 7s 6ms/step - loss: 8.6600 - accuracy: 0.1440 
    Epoch 2/15 500/500 [==============================] - 3s 6ms/step - loss: 2.1256 - accuracy: 0.3973 
    Epoch 3/15 500/500 [==============================] - 3s 6ms/step - loss: 1.5735 - accuracy: 0.5350 
    Epoch 4/15 500/500 [==============================] - 3s 6ms/step - loss: 1.2675 - accuracy: 0.6262 
    Epoch 5/15 500/500 [==============================] - 3s 6ms/step - loss: 1.0580 - accuracy: 0.6971 
    Epoch 6/15 500/500 [==============================] - 3s 6ms/step - loss: 0.9624 - accuracy: 0.7359 
    Epoch 7/15 500/500 [==============================] - 3s 6ms/step - loss: 0.8317 - accuracy: 0.7730 
    Epoch 8/15 500/500 [==============================] - 3s 6ms/step - loss: 0.7818 - accuracy: 0.7915 
    Epoch 9/15 500/500 [==============================] - 3s 6ms/step - loss: 0.6972 - accuracy: 0.8161 
    Epoch 10/15 500/500 [==============================] - 3s 6ms/step - loss: 0.6183 - accuracy: 0.8387 
    Epoch 11/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5886 - accuracy: 0.8495 
    Epoch 12/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5751 - accuracy: 0.8544 
    Epoch 13/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5609 - accuracy: 0.8620 
    Epoch 14/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5401 - accuracy: 0.8682 
    Epoch 15/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5957 - accuracy: 0.8564

**Model Summary**


Layer (type) |Output Shape |Param # 
|-|-|-|
conv2d (Conv2D) |(None, 28, 28, 36) |1008 
max_pooling2d (MaxPooling2D) |(None, 9, 9, 36) |0 
flatten (Flatten) |(None, 2916) |0 
dropout (Dropout)| (None, 2916)| 0 
dense (Dense)| (None, 2752)| 8027584 
dropout_1 (Dropout)| (None, 2752)| 0 
dense_1 (Dense)| (None, 688) |1894064 
**dropout_2 (Dropout)** |**(None, 688)**| **0** 
**dense_2 (Dense)** |**(None, 688)** |**474032** 
dense_3 (Dense) |(None, 43)| 29627

> Total params: 10,426,315 Trainable params: 10,426,315 Non-trainable params: 0

**Plot:**[ Accuracy and Loss over Epochs](https://prnt.sc/1lC6LqHC1vXU)
**Model:**  [Model Diagram](https://prnt.sc/jKZ44IXrVc0V)

## Experiment 8: Dropping the Drop
Experiment 7 showed that the additional dropout was counter-productive. I removed the additional dropout, but kept the densing effect: This seems to marginally improve the overall accuracy and reduced loss. Loss limited to 0.4 right after epoch 5 and accuracy touches 0.9 from Epoch 9th.

**Iterations**

    Epoch 1/15 2022-12-27 16:10:59.145349: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    500/500 [==============================] - 7s 6ms/step - loss: 5.3300 - accuracy: 0.4508 
    Epoch 2/15 500/500 [==============================] - 3s 6ms/step - loss: 0.8455 - accuracy: 0.7586 
    Epoch 3/15 500/500 [==============================] - 3s 6ms/step - loss: 0.6179 - accuracy: 0.8258 
    Epoch 4/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5758 - accuracy: 0.8487 
    Epoch 5/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4598 - accuracy: 0.8778 
    Epoch 6/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4528 - accuracy: 0.8776 
    Epoch 7/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3745 - accuracy: 0.8978 
    Epoch 8/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3909 - accuracy: 0.8984 
    Epoch 9/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3705 - accuracy: 0.9063 
    Epoch 10/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3598 - accuracy: 0.9130 
    Epoch 11/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3554 - accuracy: 0.9135 
    Epoch 12/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3422 - accuracy: 0.9199 
    Epoch 13/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3285 - accuracy: 0.9191 
    Epoch 14/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3635 - accuracy: 0.9152 
    Epoch 15/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3774 - accuracy: 0.9144


**Model Summary**
Layer (type) |Output Shape| Param # 
|-|-|-
conv2d (Conv2D)| (None, 28, 28, 36) |1008 
max_pooling2d (MaxPooling2D)| (None, 9, 9, 36)| 0 
flatten (Flatten) |(None, 2916)| 0 
dropout (Dropout)| (None, 2916)| 0 
dense (Dense) |(None, 2752) |8027584 
dropout_1 (Dropout) |(None, 2752) |0 
dense_1 (Dense) |(None, 688)| 1894064 
dense_2 (Dense)| (None, 688) |474032 
dense_3 (Dense) |(None, 43) |29627


> Total params: 10,426,315 Trainable params: 10,426,315 Non-trainable params: 0


**Plot:** [Accuracy and Loss over Epochs](https://prnt.sc/I_-pHlPpv-cH)

**Model:** [Model Diagram](https://prnt.sc/O0UctdB_0-nX)

# Experiment 9: Dont Drop till the very end

 -- of Hidden layers - Essentially, moved the drop that was in middle of the hidden layers, to be at just before output layer. This further ***improved accuracy to 0.95 and loss was confined to a smaller 0.2-0.3 range from Epoch 7**.*

  

**Iterations**

    Epoch 1/15 2022-12-27 16:15:41.058199: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    500/500 [==============================] - 6s 7ms/step - loss: 6.6623 - accuracy: 0.1007 
    Epoch 2/15 500/500 [==============================] - 3s 6ms/step - loss: 2.1783 - accuracy: 0.3395 
    Epoch 3/15 500/500 [==============================] - 3s 6ms/step - loss: 1.4294 - accuracy: 0.5446 
    Epoch 4/15 500/500 [==============================] - 3s 6ms/step - loss: 0.8115 - accuracy: 0.7570 
    Epoch 5/15 500/500 [==============================] - 3s 6ms/step - loss: 0.5553 - accuracy: 0.8455 
    Epoch 6/15 500/500 [==============================] - 3s 6ms/step - loss: 0.4245 - accuracy: 0.8873 
    Epoch 7/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3432 - accuracy: 0.9135 
    Epoch 8/15 500/500 [==============================] - 3s 6ms/step - loss: 0.2787 - accuracy: 0.9281 
    Epoch 9/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3178 - accuracy: 0.9267 
    Epoch 10/15 500/500 [==============================] - 3s 6ms/step - loss: 0.2130 - accuracy: 0.9478 
    Epoch 11/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3029 - accuracy: 0.9337 
    Epoch 12/15 500/500 [==============================] - 3s 6ms/step - loss: 0.2905 - accuracy: 0.9384 
    Epoch 13/15 500/500 [==============================] - 3s 6ms/step - loss: 0.2277 - accuracy: 0.9512 
    Epoch 14/15 500/500 [==============================] - 3s 6ms/step - loss: 0.3148 - accuracy: 0.9374 
    Epoch 15/15 500/500 [==============================] - 3s 6ms/step - loss: 0.2392 - accuracy: 0.9531


**Model Summary**

Layer (type)| Output Shape |Param #
|-|-|-|
conv2d (Conv2D) |(None, 28, 28, 36) |1008
max_pooling2d (MaxPooling2D)| (None, 9, 9, 36) |0
flatten (Flatten) |(None, 2916) |0
dropout (Dropout) |(None, 2916) |0
dense (Dense) |(None, 2752) |8027584
dense_1 (Dense) |(None, 688) |1894064
dense_2 (Dense) |(None, 688) |474032
**dropout_1 (Dropout)** |**(None, 688)** |**0**
dense_3 (Dense) |(None, 43) |29627


> Total params: 10,426,315 Trainable params: 10,426,315 Non-trainable params: 0
  

**Plot:** [Accuracy and Loss over Epochs](https://prnt.sc/5XqSHCOALP7k)

**Model:** [Model Diagram](https://prnt.sc/x88IZu1u9ZOy)

## Experiment 10: Batching and Sigmoid

I lastly changed my focus to the convolution layer. By tinkering around, I discovered that a convolution layer after maxpool, with sigmoid activation model and batching the fitting function gave a rather impressive outcome.  I am not entirely sure if this was truly improvement or overfitting. As early as from 5th Epoc, accuracy started touching 0.95 and loss went down to less than 0.1 by 8th Epoch. By 15th Epoch, **accuracy** was oscillating between **0.97-0.98** and **loss** was between **0.05-0.06**.

  

**Iterations**

    Epoch 1/15 2022-12-27 18:29:20.109445: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100 
    160/160 [==============================] - 7s 9ms/step - loss: 1.8089 - accuracy: 0.4847 
    Epoch 2/15 160/160 [==============================] - 1s 9ms/step - loss: 0.5114 - accuracy: 0.8363 
    Epoch 3/15 160/160 [==============================] - 1s 9ms/step - loss: 0.3119 - accuracy: 0.9007 
    Epoch 4/15 160/160 [==============================] - 2s 12ms/step - loss: 0.2010 - accuracy: 0.9336 
    Epoch 5/15 160/160 [==============================] - 1s 8ms/step - loss: 0.1380 - accuracy: 0.9548 
    Epoch 6/15 160/160 [==============================] - 1s 8ms/step - loss: 0.1085 - accuracy: 0.9651 
    Epoch 7/15 160/160 [==============================] - 1s 8ms/step - loss: 0.1018 - accuracy: 0.9680 
    Epoch 8/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0871 - accuracy: 0.9711 
    Epoch 9/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0837 - accuracy: 0.9727 
    Epoch 10/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0651 - accuracy: 0.9790 
    Epoch 11/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0492 - accuracy: 0.9842 
    Epoch 12/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0643 - accuracy: 0.9797 
    Epoch 13/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0507 - accuracy: 0.9832 
    Epoch 14/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0637 - accuracy: 0.9793 
    Epoch 15/15 160/160 [==============================] - 1s 8ms/step - loss: 0.0625 - accuracy: 0.9805

**Model Summary**

Layer (type) |Output Shape |Param #
|-|-|-
conv2d (Conv2D) |(None, 28, 28, 36) |1008
max_pooling2d (MaxPooling2D)| (None, 9, 9, 36)| 0
**conv2d_1 (Conv2D)** |**(None, 7, 7, 36)**| **11700**
flatten (Flatten) |(None, 1764) |0
dropout (Dropout) |(None, 1764) |0
dense (Dense) |(None, 2752)| 4857280
dense_1 (Dense)| (None, 688) |1894064
dense_2 (Dense)| (None, 688) |474032
dropout_1 (Dropout)|(None, 688)| 0
dense_3 (Dense)| (None, 43) |29627

  

> Total params: 7,267,711 Trainable params: 7,267,711 Non-trainable params: 0

**Plot:** [Accuracy and Loss over Epochs](https://prnt.sc/-5kl6wvZX5n6)

**Model:** [Model Diagram](https://prnt.sc/lnGhEyi4Wiz5)

# Closing Thoughts

 - Building, training and fitting models in TensorFlow is intriguing. It may seem cumbersome to get the whole thing working right in first place, but once you get it set up - There is a vast horizon for exploration. 
 - I will definitely like spending more time exploring neural networks and their applications via TensorFlow seems to have a rich ecosystem for developers and spending some time to get familiar with it is worth the investment. 
 - I look forward to enrich my theoretical and practical knowledge on the subject - The vastly ambiguous nature of the problems we can solve using these techniques seem to be strong motivators for me.  
