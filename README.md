# Python implementation of ML-PnP

### Credits

This code is based on the work from _Steffen Urban et al._ : [MLPnP - A Real-Time Maximum Likelihood Solution to the Perspective-n-Point Problem](https://arxiv.org/abs/1607.08112).

Also, this implementation is partly based on the author's Matlab [implementation](https://github.com/urbste/MLPnP_matlab).


If you use MLPnP, please cite the original paper :

    @INPROCEEDINGS {mlpnp2016,
      title={MLPNP - A REAL-TIME MAXIMUM LIKELIHOOD SOLUTION TO THE PERSPECTIVE-N-POINT PROBLEM},
      author={Urban, Steffen and Leitloff, Jens and Hinz, Stefan},
      booktitle={ISPRS Annals of Photogrammetry, Remote Sensing \& Spatial Information Sciences},
      pages={131-138},
      year={2016},
      volume={3}
    }
    
    
### Documentation

The mlpnp.py file containts all the functions to execute ML-PnP.

The function executing it is mlpnp.

The main function is provided for testing purposes :
* the algorithm randomly generates a 3D transformation from camera to world
* it samples random points in camera frame
* the transform is used to generate world coordinates (ensuring that they are in the camera FOV)
* then this set of simulated points and measurements are used in mlpnp to recover the ground truth 3D-transformation
