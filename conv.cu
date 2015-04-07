#include <assert.h>
__global__ void loop_conv(float* input, float* filters,  float* bias,
    float* output,
    int num_batches, int in_height, int in_width,
    int in_duration, int in_channels,
    int num_filters, int filt_height, int filt_width, int filt_duration,
    int filt_channels) {
    assert(filt_channels == in_channels);
    int out_height = in_height - filt_height + 1;
    int out_width = in_width - filt_width + 1;
    int out_duration = in_duration - filt_duration + 1;
    
    //# The output is H :)
    //H = np.zeros((num_batches,out_height,out_width,out_duration,num_filters))
    for (int batch_i = 0; batch_i < num_batches; batch_i++) {
      for (int out_x = 0; out_x < out_height; out_x++) { 
        for (int out_y = 0; out_y < out_width; out_y++) {
          for (int out_z = 0; out_z < out_duration; out_z++) {  
            for (int filt_i = 0; filt_i < num_filters; filt_i++) {  
              int output_flat_i = 
                batch_i * out_height * out_width * out_duration * num_filters + 
                out_x * out_width * out_duration * num_filters +
                out_y * out_duration * num_filters +
                out_z * num_filters + 
                filt_i;
              output[output_flat_i] += bias[filt_i];
              for (int filt_x = 0; filt_x < filt_height; filt_x++) {  
                for (int filt_y = 0; filt_y < filt_width; filt_y++) {  
                  for (int filt_z = 0; filt_z < filt_duration; filt_z++) {  
                    for (int filt_chan_i = 0; filt_chan_i < filt_channels; filt_chan_i++) {  
                      int filter_flat_i = 
                          filt_i * filt_height * filt_width * filt_duration * filt_channels +
                          filt_x * filt_width *filt_duration * filt_channels +
                          filt_y *filt_duration*filt_channels +
                          filt_z * filt_channels
                          + filt_chan_i;
                      float weight = filters[filter_flat_i];
                      int input_flat_i = 
                          batch_i * in_height * in_width * in_duration * in_channels + 
                          (out_x + filt_x) * in_width * in_duration * in_channels +
                          (out_y + filt_y) * in_duration * in_channels +
                          (out_z + filt_z) * in_channels +
                          filt_chan_i;
                      float input_value = input[input_flat_i];
                      
                      output[output_flat_i] += weight * input_value;
                      /*
                        weight = W[filt_i, filt_x, filt_y, filt_z, filt_chan_i]
                        input_val =  X[batch_i, out_x + filt_x, out_y + filt_y, out_z + filt_z, filt_chan_i]
                        H[batch_i, out_x, out_y, out_z, filt_i] += \
                             weight * input_val
                             */
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
}