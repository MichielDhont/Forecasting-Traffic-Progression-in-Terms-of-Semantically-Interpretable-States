# Forecasting Traffic Progression in Terms of Semantically Interpretable States by Exploring Multiple Data Representations
[Michiel Dhont](https://orcid.org/0000-0002-5679-0991), [Adrian Munteanu](https://orcid.org/0000-0001-7290-0428), [Elena Tsiporkova](https://orcid.org/0009-0003-7202-3471)

In this repository, some supportive material for the article "Forecasting Traffic Progression in Terms of Semantically Interpretable States by Exploring Multiple Data Representations" is given. Once the article is accepted and published, a direct link to the article will be provided here.


## Training Process

The models were trained using TensorFlow with the settings mentioned below. Note that during experimenting, several settings (learning rate, optimiser, objective function, ...) were tested to find the most optimal configuration based on the learning curves of training and test data.

###	Callbacks
- ModelCheckpoint: The model weights were saved when an improvement in validation loss was observed, ensuring the best model was retained.
- ReduceLROnPlateau: The learning rate was reduced by a factor of 0.1 if the validation loss did not improve for 30 consecutive epochs, with a maximum learning rate set to 0.01 and a minimum learning rate set to 1e-9.

###	Training
- The objective function is the categorical cross-entropy, while the Adam optimizer is used for adjusting the model's weight during training.
- The model was trained for 300 epochs using the training dataset.
- Validation was performed on a separate validation dataset at the end of each epoch.
- When training a model for a further forecast horizon, the weights from the model trained on the preceding forecast horizon are used to initialize the new model.

These settings helped in optimizing the model performance by saving the best weights and adjusting the learning rate dynamically.


## Model Architectures
During the research phase, a several model architectures and configurations were tested. The most interesting/promising architectures are given below and their results are discussed in the article.


### CNN - Temporal Data Representation as Input
<img title="CNN_temporal" alt="CNN_temporal" src="/figures/CNN_temporal.png">

```{python}
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 60, 3, 1)]        0         
                                                                 
 conv2d (Conv2D)             (None, 60, 3, 16)         160       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 30, 2, 16)        0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 30, 2, 32)         4640      
                                                                 
 conv1d (Conv1D)             (None, 30, 2, 64)         6208      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 15, 1, 64)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 960)               0         
                                                                 
 dense (Dense)               (None, 128)               123008    
                                                                 
 y1_output (Dense)           (None, 6)                 774       
                                                                 
=================================================================
Total params: 134,790
Trainable params: 134,790
Non-trainable params: 0
_________________________________________________________________
```

```{python}
{'name': 'model',
 'layers': [{'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 60, 3, 1),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'name': 'input_1',
   'inbound_nodes': []},
  {'class_name': 'Conv2D',
   'config': {'name': 'conv2d',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'conv2d',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 1),
    'data_format': 'channels_last'},
   'name': 'max_pooling2d',
   'inbound_nodes': [[['conv2d', 0, 0, {}]]]},
  {'class_name': 'Conv2D',
   'config': {'name': 'conv2d_1',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'conv2d_1',
   'inbound_nodes': [[['max_pooling2d', 0, 0, {}]]]},
  {'class_name': 'Conv1D',
   'config': {'name': 'conv1d',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (3,),
    'strides': (1,),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1,),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'conv1d',
   'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]},
  {'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d_1',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'name': 'max_pooling2d_1',
   'inbound_nodes': [[['conv1d', 0, 0, {}]]]},
  {'class_name': 'Flatten',
   'config': {'name': 'flatten',
    'trainable': True,
    'dtype': 'float32',
    'data_format': 'channels_last'},
   'name': 'flatten',
   'inbound_nodes': [[['max_pooling2d_1', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'dense',
    'trainable': True,
    'dtype': 'float32',
    'units': 128,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'dense',
   'inbound_nodes': [[['flatten', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'y1_output',
    'trainable': True,
    'dtype': 'float32',
    'units': 6,
    'activation': 'softmax',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'y1_output',
   'inbound_nodes': [[['dense', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['y1_output', 0, 0]]}
```


### CNN - Time-Frequency Data Representation as Input
<img title="CNN_timefrequency" alt="CNN_timefrequency" src="/figures/CNN_timefrequency.png">


```{python}
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 8, 8, 3)]         0         
                                                                 
 conv2d (Conv2D)             (None, 8, 8, 16)          448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 4, 4, 16)         0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 4, 4, 32)          4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 2, 2, 32)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 2, 2, 64)          18496     
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 1, 1, 64)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 128)               8320      
                                                                 
 y1_output (Dense)           (None, 6)                 774       
                                                                 
=================================================================
Total params: 32,678
Trainable params: 32,678
Non-trainable params: 0
_________________________________________________________________
```

```{python}
{'name': 'model',
 'layers': [{'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 8, 8, 3),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'name': 'input_1',
   'inbound_nodes': []},
  {'class_name': 'Conv2D',
   'config': {'name': 'conv2d',
    'trainable': True,
    'dtype': 'float32',
    'filters': 16,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'conv2d',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'name': 'max_pooling2d',
   'inbound_nodes': [[['conv2d', 0, 0, {}]]]},
  {'class_name': 'Conv2D',
   'config': {'name': 'conv2d_1',
    'trainable': True,
    'dtype': 'float32',
    'filters': 32,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'conv2d_1',
   'inbound_nodes': [[['max_pooling2d', 0, 0, {}]]]},
  {'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d_1',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'name': 'max_pooling2d_1',
   'inbound_nodes': [[['conv2d_1', 0, 0, {}]]]},
  {'class_name': 'Conv2D',
   'config': {'name': 'conv2d_2',
    'trainable': True,
    'dtype': 'float32',
    'filters': 64,
    'kernel_size': (3, 3),
    'strides': (1, 1),
    'padding': 'same',
    'data_format': 'channels_last',
    'dilation_rate': (1, 1),
    'groups': 1,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'conv2d_2',
   'inbound_nodes': [[['max_pooling2d_1', 0, 0, {}]]]},
  {'class_name': 'MaxPooling2D',
   'config': {'name': 'max_pooling2d_2',
    'trainable': True,
    'dtype': 'float32',
    'pool_size': (2, 2),
    'padding': 'valid',
    'strides': (2, 2),
    'data_format': 'channels_last'},
   'name': 'max_pooling2d_2',
   'inbound_nodes': [[['conv2d_2', 0, 0, {}]]]},
  {'class_name': 'Flatten',
   'config': {'name': 'flatten',
    'trainable': True,
    'dtype': 'float32',
    'data_format': 'channels_last'},
   'name': 'flatten',
   'inbound_nodes': [[['max_pooling2d_2', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'dense',
    'trainable': True,
    'dtype': 'float32',
    'units': 128,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'dense',
   'inbound_nodes': [[['flatten', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'y1_output',
    'trainable': True,
    'dtype': 'float32',
    'units': 6,
    'activation': 'softmax',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'y1_output',
   'inbound_nodes': [[['dense', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['y1_output', 0, 0]]}
```


### RNN (GRU) - Temporal Data Representation as Input
<img title="RNN_temporal" alt="RNN_temporal" src="/figures/RNN_temporal.png">

```{python}
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 120, 3)]          0         
                                                                 
 gru (GRU)                   (None, 120, 64)           13248     
                                                                 
 gru_1 (GRU)                 (None, 64)                24960     
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 32)                2080      
                                                                 
 y1_output (Dense)           (None, 6)                 198       
                                                                 
=================================================================
Total params: 40,486
Trainable params: 40,486
Non-trainable params: 0
_________________________________________________________________
```

```{python}
{'name': 'model',
 'layers': [{'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 120, 3),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'name': 'input_1',
   'inbound_nodes': []},
  {'class_name': 'GRU',
   'config': {'name': 'gru',
    'trainable': True,
    'dtype': 'float32',
    'return_sequences': True,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'time_major': False,
    'units': 64,
    'activation': 'relu',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'shared_object_id': 1},
    'recurrent_initializer': {'class_name': 'Orthogonal',
     'config': {'gain': 1.0, 'seed': None},
     'shared_object_id': 2},
    'bias_initializer': {'class_name': 'Zeros',
     'config': {},
     'shared_object_id': 3},
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'reset_after': True},
   'name': 'gru',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'class_name': 'GRU',
   'config': {'name': 'gru_1',
    'trainable': True,
    'dtype': 'float32',
    'return_sequences': False,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'time_major': False,
    'units': 64,
    'activation': 'relu',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'shared_object_id': 6},
    'recurrent_initializer': {'class_name': 'Orthogonal',
     'config': {'gain': 1.0, 'seed': None},
     'shared_object_id': 7},
    'bias_initializer': {'class_name': 'Zeros',
     'config': {},
     'shared_object_id': 8},
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'reset_after': True},
   'name': 'gru_1',
   'inbound_nodes': [[['gru', 0, 0, {}]]]},
  {'class_name': 'Dropout',
   'config': {'name': 'dropout',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'name': 'dropout',
   'inbound_nodes': [[['gru_1', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'dense',
    'trainable': True,
    'dtype': 'float32',
    'units': 32,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'dense',
   'inbound_nodes': [[['dropout', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'y1_output',
    'trainable': True,
    'dtype': 'float32',
    'units': 6,
    'activation': 'softmax',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'y1_output',
   'inbound_nodes': [[['dense', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['y1_output', 0, 0]]}
```


### RNN (GRU) - Symbolic Data Representation as Input
<img title="RNN_symbolic" alt="RNN_symbolic" src="/figures/RNN_symbolic.png">

```{python}
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 8, 6)]            0         
                                                                 
 gru (GRU)                   (None, 8, 64)             13824     
                                                                 
 gru_1 (GRU)                 (None, 64)                24960     
                                                                 
 dropout (Dropout)           (None, 64)                0         
                                                                 
 dense (Dense)               (None, 32)                2080      
                                                                 
 y1_output (Dense)           (None, 6)                 198       
                                                                 
=================================================================
Total params: 41,062
Trainable params: 41,062
Non-trainable params: 0
_________________________________________________________________
```

```{python}
{'name': 'model',
 'layers': [{'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 8, 6),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'name': 'input_1',
   'inbound_nodes': []},
  {'class_name': 'GRU',
   'config': {'name': 'gru',
    'trainable': True,
    'dtype': 'float32',
    'return_sequences': True,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'time_major': False,
    'units': 64,
    'activation': 'relu',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'shared_object_id': 1},
    'recurrent_initializer': {'class_name': 'Orthogonal',
     'config': {'gain': 1.0, 'seed': None},
     'shared_object_id': 2},
    'bias_initializer': {'class_name': 'Zeros',
     'config': {},
     'shared_object_id': 3},
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'reset_after': True},
   'name': 'gru',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'class_name': 'GRU',
   'config': {'name': 'gru_1',
    'trainable': True,
    'dtype': 'float32',
    'return_sequences': False,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'time_major': False,
    'units': 64,
    'activation': 'relu',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'shared_object_id': 6},
    'recurrent_initializer': {'class_name': 'Orthogonal',
     'config': {'gain': 1.0, 'seed': None},
     'shared_object_id': 7},
    'bias_initializer': {'class_name': 'Zeros',
     'config': {},
     'shared_object_id': 8},
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'reset_after': True},
   'name': 'gru_1',
   'inbound_nodes': [[['gru', 0, 0, {}]]]},
  {'class_name': 'Dropout',
   'config': {'name': 'dropout',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'name': 'dropout',
   'inbound_nodes': [[['gru_1', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'dense',
    'trainable': True,
    'dtype': 'float32',
    'units': 32,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'dense',
   'inbound_nodes': [[['dropout', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'y1_output',
    'trainable': True,
    'dtype': 'float32',
    'units': 6,
    'activation': 'softmax',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'y1_output',
   'inbound_nodes': [[['dense', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['y1_output', 0, 0]]}
```


### RNN (GRU) - Time-Frequency Data Representation as Input
<img title="RNN_timefrequency" alt="RNN_timefrequency" src="/figures/RNN_timefrequency.png">

```{python}
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 8, 24)]           0         
                                                                 
 gru_2 (GRU)                 (None, 8, 64)             17280     
                                                                 
 gru_3 (GRU)                 (None, 64)                24960     
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 y1_output (Dense)           (None, 6)                 198       
                                                                 
=================================================================
Total params: 44,518
Trainable params: 44,518
Non-trainable params: 0
_________________________________________________________________
```

```{python}
{'name': 'model',
 'layers': [{'class_name': 'InputLayer',
   'config': {'batch_input_shape': (None, 8, 24),
    'dtype': 'float32',
    'sparse': False,
    'ragged': False,
    'name': 'input_1'},
   'name': 'input_1',
   'inbound_nodes': []},
  {'class_name': 'GRU',
   'config': {'name': 'gru',
    'trainable': True,
    'dtype': 'float32',
    'return_sequences': True,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'time_major': False,
    'units': 64,
    'activation': 'relu',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'shared_object_id': 1},
    'recurrent_initializer': {'class_name': 'Orthogonal',
     'config': {'gain': 1.0, 'seed': None},
     'shared_object_id': 2},
    'bias_initializer': {'class_name': 'Zeros',
     'config': {},
     'shared_object_id': 3},
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'reset_after': True},
   'name': 'gru',
   'inbound_nodes': [[['input_1', 0, 0, {}]]]},
  {'class_name': 'GRU',
   'config': {'name': 'gru_1',
    'trainable': True,
    'dtype': 'float32',
    'return_sequences': False,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'time_major': False,
    'units': 64,
    'activation': 'relu',
    'recurrent_activation': 'sigmoid',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None},
     'shared_object_id': 6},
    'recurrent_initializer': {'class_name': 'Orthogonal',
     'config': {'gain': 1.0, 'seed': None},
     'shared_object_id': 7},
    'bias_initializer': {'class_name': 'Zeros',
     'config': {},
     'shared_object_id': 8},
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'implementation': 2,
    'reset_after': True},
   'name': 'gru_1',
   'inbound_nodes': [[['gru', 0, 0, {}]]]},
  {'class_name': 'Dropout',
   'config': {'name': 'dropout',
    'trainable': True,
    'dtype': 'float32',
    'rate': 0.2,
    'noise_shape': None,
    'seed': None},
   'name': 'dropout',
   'inbound_nodes': [[['gru_1', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'dense',
    'trainable': True,
    'dtype': 'float32',
    'units': 32,
    'activation': 'relu',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'dense',
   'inbound_nodes': [[['dropout', 0, 0, {}]]]},
  {'class_name': 'Dense',
   'config': {'name': 'y1_output',
    'trainable': True,
    'dtype': 'float32',
    'units': 6,
    'activation': 'softmax',
    'use_bias': True,
    'kernel_initializer': {'class_name': 'GlorotUniform',
     'config': {'seed': None}},
    'bias_initializer': {'class_name': 'Zeros', 'config': {}},
    'kernel_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'bias_constraint': None},
   'name': 'y1_output',
   'inbound_nodes': [[['dense', 0, 0, {}]]]}],
 'input_layers': [['input_1', 0, 0]],
 'output_layers': [['y1_output', 0, 0]]}
```
