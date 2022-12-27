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

## Experiment 8: Cutting back Dropouts
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
