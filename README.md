# pylearn3dconv
Volumetric/3d Convolutions for pylearn using cudnn

See test/test_training.py for usage.
For using the 3d FFT convolution, use Theano3dConv with
`THEANO_FLAGS=optimizer_including=conv3d_fft:convgrad3d_fft:convtransp3d_fft`
(see http://deeplearning.net/software/theano/library/tensor/nnet/conv.html)

TODO: Proper usage example :)
