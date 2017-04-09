#include <algorithm>
#include <vector>

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"
//see: http://blog.csdn.net/liyaohhh/article/details/52115638
namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //softmax_axis主要是指定从哪个维度开始切，默认是1. 区分了outer_num_和inner_num_
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);// 可以看到所谓的Reshape函数真正Reshape的只有这一句话，其他语句做了其他事
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_)); //构造数量为1放入值为bottom[0]->shape(softmax_axis_)的vector
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data(); 
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data); ///sum_multiplier_这里都是1，用于辅助计算，可以看作一个行向量，或者行数为1的矩阵
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);//可见softmax_axis_是channel的axis
  int dim = bottom[0]->count() / outer_num_; //外围的维度count（累乘是多少）
  caffe_copy(bottom[0]->count(), bottom_data, top_data); //bottom的数据先copy至top
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);//bottom_data[i * dim] copyto scale_data
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }//至此scale_data存储的都是遍历inner_num_中的bottom最大值
    // subtraction, see http://blog.csdn.net/bailufeiyan/article/details/50879391
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
        //分析一下:C=alpha*A*B+beta*C ,这里alpha=-1,A=sum_multiplier_.cpu_data(),B=scale_data,beta=1.0,C=top_data
        //所以top_data=(-1)*sum_multiplier_.cpu_data()*scale_data+1.0*top_data, 这里sum_multiplier是辅助计算的，值都是1
        //先求取max, 然后所有值先减去了这个max，目的作者也给了注释是数值问题，毕竟之后是要接上e为底的指数运算的，所以值不可以太大，这个操作相当合理。
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp， y=alpha*A*x+beta*y ，这里scale_data = top_data * sum_multiplier_.cpu_data() + 0 * scale_data
    //因为sum_multiplier_的每个元素值都是1，是辅助计算的，所以scale_data最后变成了top_data的sum之和
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      //top_data[i] = top_data[i] / scale_data[i] 而scale_data就是上一步计算的sum
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_; //注意：这样top_data每隔inner_num_出现一次softmax值，
    }
  }
}

//see：https://www.zhihu.com/question/28927103。仍然不太明白outer_num_和inner_num_是干嘛的？
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    //此处计算点积，注意到top_diff已经拷贝到bottom_diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction， C=alpha*A*B+beta*C 
    /**
    K=-1,K是A的列数和B的行数
    alpha=-1.0, A=sum_multiplier_.cpu_data(), B=scale_data, beta=1.0, C = bottom_diff + i * dim
    所以bottom_diff[i * dim]= -1.0*(sum_multiplier_.cpu_data() * scale_data) + bottom_diff[i * dim]
    //此处为计算大括号内的减法：https://www.zhihu.com/question/28927103
    **/
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication: bottom_diff[i] = bottom_diff[i] * top_data[i]
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
