#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

/***
dataLayer作为整个网络的输入层，数据从leveldb中取。leveldb的数据是通过图片转换过来的。
网络建立的时候，datalayer主要是负责设置一些参数，比如batchsize，channels，height，width等。通过读leveldb一个数据块来获取这些信息。
然后启动一个线程来预先从leveldb拉取一批数据，这些数据是图像数据和图像标签
正向传播的时候，datalayer就把预先拉取好数据拷贝到指定的cpu或者gpu的内存。
然后启动新线程再预先拉取数据，这些数据留到下一次正向传播使用
***/

namespace caffe {

template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);

  DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
