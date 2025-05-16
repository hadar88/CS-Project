# `make_dataset.py`

This script contains the function `make_xs` that gives the initial data of the menus.
It also contains the function `make_labels` that gives the menu data of the menus.
[View Script](./make_dataset.py)

# `menu_output_transform.py`

This script contains the function `menu_dict_to_tensor` that convert the menu dictionary to a tensor.
It also contains the function `menu_tensor_to_dict` that convert the menu tensor to a dictionary.
It also contains the function `transform` that transform the menu to the menu data.
[View Script](./menu_output_transform.py)

# `model.py`

This strip builds the model.
To train the model use -> python3 model.py --split train
To test the model use -> python3 model.py --split test
To validate the model use -> python3 model.py --split val
To use tensorboard use -> tensorboard --logdir=runs/model_v{MODEL_VERSION}
[View Script](./model.py)


 


 