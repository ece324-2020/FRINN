# FRINN

Fruit Ripeness Identification using Neural Networks

The goal of FRINN is to develop a model (NN) to classify both the state and type of an input fruit image. The type of fruits are apples, oranges, bananas, raspberries, strawberries and the state of fruits are ripe, unripe and rotten.

Structure of Repository:

- Baseline Folder: 
  - Baseline model script consisting of two random forests (one for state prediction and one for type prediction) 
- Data_FRINN Folder: 
  - Folder consisting of 15 subfolders containing the training data (png images)
- Data processing Folder:
  - Custom data loader: File initially created for splitting up labels into state and type, not actually used as not efficient
  - Data science for FRINN: Training data visualization and plotting of distribution of classes
  - Data augmentation: script to augment collected dataset using 9 data augmentation techniques
- Transfer Learning Folder:
  - Transfer_learning_one_model : final transfer learning notebook for single head model on 15 labels/classes
  - Transfer_learning_conv_layer : final transfer learning notebook for multi head model (1 additional trainable conv layer)
  - Transfer_learning_noconv_layer : final transfer learning notebook for multi head model (no additional trainable conv layer)
- Multi head Folder:
  - Multi head cnn rough work: first initial notebook with working training loop for models
  - Multi head cnn final: final non transfer learning notebook for multi head model
          
