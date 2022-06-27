def binary_FFNN_model(encoder: Model,
                      input_shape,
                      hidden_layers: list,
                      hid_layers_act: str = 'ReLU',
                      outp_layer_act: str = 'sigmoid',
                      optimizer : Optimizer = Adam(learning_rate=.01),
                      loss: Loss = BinaryCrossentropy(),
                      metrs: list = [
                                        BinaryAccuracy(),
                                        Precision(),
                                        Recall(),
                                        AUC(),
                                        TruePositives(), 
                                        TrueNegatives(),
                                        FalsePositives(),
                                        FalseNegatives()
                                      ],
                      
                      ) -> Model:
  """ 
  Build the structure of the ffnn model
  Parameters:
    - encoder: a pre trained encoder which filters out the images
    - input_shape: the number of input that the model must handle
    - hidden_layers: an iterator containing the amount of neurons in each hidden layer
    - hid_layers_act: the activation function of the neurons in the hidden layers,
    - outp_layer_act: the activation function of the neurons in the output layer,
    - optimizer : The optimizer that will be used,
    - metrs : the list of metrics used to evaluate the model's performance
  Return:
    The compiled model
  """

  # Definition of the input and output (dense) layer
  input_layer = Input(shape = input_shape)
  if encoder:
    for k,v in encoder._get_trainable_state().items():
      k.trainable = False
    pre = encoder(input_layer)
  else:
    pre = Flatten()(input_layer)
  
  hidden = Rescaling(1./255)(pre)
  
  for i in hidden_layers:
    # hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Dropout(.3)(hidden)
    hidden = Dense(i, activation = hid_layers_act)(hidden)
    
  output_layer = Dense(1, activation = outp_layer_act)
  ffnn = Model(inputs = input_layer, outputs = output_layer(hidden))


  ffnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs,
        run_eagerly=True
    )
  return ffnn
