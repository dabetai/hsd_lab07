#include "fpga_api.h"
#include <stdio.h>
#include <iostream>
#include <cstring>

using namespace std;

#define min(x, y) (((x) < (y)) ? (x) : (y))

FPGA::FPGA(off_t data_addr, off_t output_addr, int m_size, int v_size)
{
  m_size_ = m_size;
  v_size_ = v_size;
  data_size_ = (m_size_ + 1) * v_size_; // fpga bram data size

  output_ = new unsigned int[m_size_]; // use output_ as tempolar output
  data_ = new float[data_size_];

  num_block_call_ = 0;
}
FPGA::~FPGA()
{
  delete[] output_;
  delete[] data_;
}

float *FPGA::matrix(void)
{
  return data_ + v_size_;
}

float *FPGA::vector(void)
{
  return data_;
}

void FPGA::reset(void)
{
  num_block_call_ = 0;
}

int FPGA::num_block_call(void)
{
  return num_block_call_;
}

const float *FPGA::blockMV()
{
  num_block_call_ += 1;

  // cpu version
  float *vec = this->vector();
  float *mat = this->matrix();
  float *out = reinterpret_cast<float *>(output_);

  for (int i = 0; i < m_size_; ++i)
  {
    out[i] = 0;
    for (int j = 0; j < v_size_; ++j)
      out[i] += vec[j] * mat[v_size_ * i + j];
  }

  for (int i = 0; i < m_size_; ++i)
    data_[i] = out[i];

  return data_;
}

void FPGA::largeMV(const float *large_mat, const float *input, float *output, int num_input, int num_output)
{
  float *vec = this->vector();
  float *mat = this->matrix();

  // 0) Initialize output vector
  for (int i = 0; i < num_output; ++i)
    output[i] = 0;

  for (int i = 0; i < num_output; i += m_size_)
  {
    for (int j = 0; j < num_input; j += v_size_)
    {
      // 0) Initialize input vector
      int block_row = min(m_size_, num_output - i);
      int block_col = min(v_size_, num_input - j);

      // 1) Assign a vector
      memset(vec, 0, v_size_);
      memcpy(vec, input + j, sizeof(float) * block_col);

      // 2) Assign a matrix
      // IMPLEMENT THIS
      memset(mat, 0, data_size_ - v_size_);
      for(int k = i; k < i + block_row; ++k) {
	      memcpy(mat + ((k-i) * v_size_), large_mat + ((k * num_input) + j), sizeof(float) * block_col);
      }

      // 3) Call a function `blockMV() to execute MV multiplication
      const float *ret = this->blockMV();

      // 4) Accumulate intermediate results
      for (int row = 0; row < block_row; ++row)
        output[i + row] += ret[row];
    }
  }
}

void FPGA::convLowering(const std::vector<std::vector<std::vector<std::vector<float>>>> &cnn_weights,
                        std::vector<std::vector<float>> &new_weights,
                        const std::vector<std::vector<std::vector<float>>> &inputs,
                        std::vector<std::vector<float>> &new_inputs)
{
  /*
   * Arguments:
   *
   * conv_weights: [conv_channel, input_channel, conv_height, conv_width]
   * new_weights: [new_weights_height, new_weight_width]
   * inputs: [input_channel, input_height, input_width]
   * new_inputs: [new_inputs_height, new_inputs_width]
   *
   */

  int conv_channel = cnn_weights.size();
  int input_channel = cnn_weights[0].size();
  int conv_height = cnn_weights[0][0].size();
  int conv_width = cnn_weights[0][0][0].size();
  //int input_channel = inputs.size();
  int input_height = inputs[0].size();
  int input_width = inputs[0][0].size();

  // IMPLEMENT THIS
  // For example,
  // new_weights[0][0] = cnn_weights[0][0][0][0];
  // new_inputs[0][0] = inputs[0][0][0];

  int new_weights_height = conv_channel;
  int new_weights_width = input_channel * conv_height * conv_width;
  int new_inputs_height = input_channel * conv_height * conv_width;
  int new_inputs_width = (input_width - conv_width + 1) * (input_height - conv_height + 1);
  
  int conv_size = conv_height * conv_width;
  int distance_height = input_height - conv_height + 1;
  int distance_width = input_width - conv_width + 1;

  int new_inputs_i;
  int new_inputs_j = -1;
  for(int filter_i = 0; filter_i < distance_height; filter_i ++) {
    for(int filter_j = 0; filter_j < distance_width; filter_j ++) {
      new_inputs_i = 0;
      new_inputs_j ++;
      for(int chan = 0; chan < input_channel; chan ++) {
        for(int conv_i = 0; conv_i < conv_height; conv_i ++) {
          for(int conv_j = 0; conv_j < conv_width; conv_j ++) {
            new_inputs[new_inputs_i ++][new_inputs_j] = inputs[chan][filter_i + conv_i][filter_j + conv_j];
            printf("new_inputs[%d][%d] = inputs[%d][%d][%d]\n", new_inputs_i - 1, new_inputs_j, chan, filter_i + conv_i, filter_j + conv_j);
          }
        }
      }
    }
  }

  int new_weights_i = -1;
  int new_weights_j;
  for(int c_chan = 0; c_chan < conv_channel; c_chan ++) {
    new_weights_i ++;
    new_weights_j = 0;
    for(int i_chan = 0; i_chan < input_channel; i_chan ++) {
      for(int conv_i = 0; conv_i < conv_height; conv_i ++) {
        for(int conv_j = 0; conv_j < conv_width; conv_j ++) {
          new_weights[new_weights_i][new_weights_j ++] = cnn_weights[c_chan][i_chan][conv_i][conv_j];
          printf("new_weights[%d][%d] = cnn_weights[%d][%d][%d]\n", new_weights_i, new_weights_j - 1, c_chan, i_chan, conv_i, conv_j);
        }
      }        
    }
  }
  
}
