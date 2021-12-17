# Wine Quality Prediction

## Packages

Python 3

Pandas

Tensorflow 2.6

Numpy

Sklearn

## How to run models

### Run KNN or SVM

Run

```bash
python SVM_and_KNN_demo.py
```

some configurations are as following:

"removed_field": attributes to be removed,

"use_imbalanced": whether to use AllKNN to solve data imbalance,

"im_n_neighbors": "n_neighbors" parameter of AllKNN,

"n_neighbors": "n_neighbors" parameter of KNN,

"class_weight": "class_weight" parameter of SVM(default: {0: 1, 1: 1}).

### Run ANN

Run

```
python ANN_demo.py
```


1. To run demo.py file to see the results of 
   *multi-classification with input parameters: model_type = 'mul', wine = 'Red' or 'White'
   *binary classification with input parameters: model_type = 'binary', wine = 'Red' or 'White'

2. Notice: Do not change the name of .h5 files in the Model, which are different weight files for each model.

3. Notice: If you want to run training functions in the Train dectionary, plase change the name of .h5 files, as following shown:
  
     ```python
     if mode == 'White':
          model.save("../Model/your_weight_w.h5")
     else:
          model.save("../Model/your_weight_r.h5")
     ```
     
     
     
4. Notice: Delete Layer that name='L2' in the MyNet.py for training of multi-classification model.
   
   ```python
   x = Dense(110, activation='relu', name='L1', kernel_initializer=kernel, bias_initializer=bias)(input_data)
   
   x = LayerNormalization(name='Nor_L1')(x)
   
   x = Dense(110, activation='relu', name='L2'
                     , kernel_initializer=kernel, bias_initializer=bias)(x)
   ```
## How to run data analysis

Run

```
python redWineSNS.py
python whiteWineSNS.py
```

in utils folder to get attribute analysis of red and white wine dataset.
