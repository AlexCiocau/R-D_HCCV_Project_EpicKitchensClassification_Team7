# R-D_HCCV_Project_EpicKitchensClassification_Team7
AI model for video classification of EpicKitchens dataset by Alexandru Ciocau, Joren Van den Vondel, Maarten Luypaert

## Project Motivation:
    The goal of the project is to perform video classification on the EPIC-KITCHENS dataset: identify actions performed and items used.

    To achieve this goal multiple solutions were explored and hence multiple models were trained:
        1. 3D_CNN (untrained) = a simple 3-layer 3D CNN which is considered to obtain a performance baseline
        2. Deep_3D_CNN (untrained) =  10-layer 3D CNN to asses performance improvement given by increased depth
        3. MC3-18_CNN (pretrained) = deep 18-layer 3D CNN which is pretrained and only "familiarized" with the data

## Description:

### Folders:
    3D_CNN, Deep_3D_CNN and MC3_CNN contain their respective model files as dataloading and preprocessing differs from one model to the other

### General files: 

    main.py = puts everything together: instantiates dataset, dataloader and NN; trains created net for K epochs; saves model
    dataloader.py = used to define the custom dataset class; in pytorch in order to use your own dataset and take advantage of the dataloader (which optimizes the input of data into   the model), such a class has to be created 
    3D_CNN.py = definition of NN class
    preprocess.py = used to preprocess the data into tensors ready to be passed to the NN for training (otherwise for each epoch the training videos would have to be converted into tensors resulting in excessive computation)
    training.py = loads the model and performs additional epochs of training if more training is desired; it was developed so that the model must not be trained all at once (300 epochs or so takes a very long time)
    validation.py = the trained model is tested on the testing dataset to obtain testing accuracy
    