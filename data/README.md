# Instructions for the data



Here we provide model checkpoints used in manuscript and additional data needed for network topography.
One need to change config files to provide appropriate data path.


## Description of the data
Data used in validation and obtaining results described in paper is provided on mendeley. Data is already preprocessed.
In every RESULTS_XXX subfolder there are .npy files with FEM data for every artery in CoW.

If One want to use their own data for in verse it shoud have following form:
The format of our multi-input-output dataset (MIODataset) should be as follows:

```python
Dataset = [
    [X1, Y1, Theta1, Inputs_funcs1],
    [X2, Y2, Theta2, Inputs_funcs2],
   ...
]
```
- **X**: (N x N_in) numpy array, representing input mesh points
    - N: number of points
    - N_in: input spatial dimension
- **Y**: (N x N_out) numpy array, representing physical fields defined on these mesh points
    - N_out: output dimension, N_out must be at least 1, shape (N,) is not allowed
- **Theta**: (N_theta,) numpy array, global parameters for this sample
    - N_theta: dimension of global parameters
- **Input_funcs**: `tuple (inputs_1, ..., inputs_n)`，every `inputs_i` is a numary array of shape (N_i, f_i), it can be (None,) :
    - N_i: number of points used to discretize this input function
    - f_i: dimension of this input function plus the dimension of geometry, actually it is the concat of (x_i, f(x_i)).
    
- **Note:**
    - For a single sample, The number of points must match, i.e, ``X.shape[0]=Y.shape[0]``, but it can vary with samples
    - For global parameters, the dimension must be the same across all samples



## Instructions to obtain the data
Validation data will be made avaliable on Mendeley. Training and Testing data will be made avaliable by corresponding author upon resonable request, since files are too large for mendeley storage.

## Instructions to process the data
Processing functions are abaliable in data_utils.py file.
