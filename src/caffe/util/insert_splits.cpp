#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "caffe/common.hpp"
#include "caffe/util/insert_splits.hpp"

namespace caffe {

void InsertSplits(const NetParameter& param, NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  //将一个top对应多个bottom的情况改为
  //将该top与每个bottom中间连线上加一个split layer 也即
  //top -> bottom1   top -> bottom2
  //改为 top -> split1  -> bottom1   top -> split2 ->  bottom2
  //想象一下链表插入节点
  param_split->CopyFrom(param);
  param_split->clear_layer();
  map<string, pair<int, int> > blob_name_to_last_top_idx;
  map<pair<int, int>, pair<int, int> > bottom_idx_to_source_top_idx;
  map<pair<int, int>, int> top_idx_to_bottom_count;
  map<pair<int, int>, float> top_idx_to_loss_weight;
  map<pair<int, int>, int> top_idx_to_bottom_split_idx;
  map<int, string> layer_idx_to_layer_name;
  for (int i = 0; i < param.layer_size(); ++i) {//遍历一个layer
    const LayerParameter& layer_param = param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();//将  index：name  映射保存到layer_idx_to_layer_name map中
    for (int j = 0; j < layer_param.bottom_size(); ++j) {//遍历该layer中的bottom，每个layer中都可以定义多个top和bottom
      const string& blob_name = layer_param.bottom(j);//取一个bottom的name，注意每个bottom对应一个blob
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {//如果这个bottom没有对应的top供应数据，报错
        LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
                   << layer_param.name() << "', bottom index " << j << ")";
      }
      const pair<int, int>& bottom_idx = make_pair(i, j);//第i层 第j个bottom
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];//找到对应的哪一层的哪个top，注意这里返回的是个pair<int, int>
      bottom_idx_to_source_top_idx[bottom_idx] = top_idx;//将  pair映射到pair， 也即哪一层的哪个bottom对应到哪一层的哪个top
      ++top_idx_to_bottom_count[top_idx];//记录下该top对应的bottom数量
    }
    for (int j = 0; j < layer_param.top_size(); ++j) {
      //由于对应的top和bottom的名字是相同的，这里将top名与top保存，这里需要理解一下。这样当要找一个bottom对应的top的时候就能找到这个同名的top
      const string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
    }
    // A use of a top blob as a loss should be handled similarly to the use of
    // a top blob as a bottom blob to another layer.
    //这里loss weight参数用的少，不是很理解这里将top对应到一个top的操作
    const int last_loss =
        std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const string& blob_name = layer_param.top(j);
      const pair<int, int>& top_idx = blob_name_to_last_top_idx[blob_name];
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      if (top_idx_to_loss_weight[top_idx]) {
        ++top_idx_to_bottom_count[top_idx];
      }
    }
  }
  //到这里主要完成了统计各个层的信息，将bottom与top对应，会有一个top对多个bottom的情况，这时一个top一个blob已经不好了，需要对应每个bottom添加blob

  for (int i = 0; i < param.layer_size(); ++i) {
    LayerParameter* layer_param = param_split->add_layer();//先加入该层，后面根据需要insert其他层
    layer_param->CopyFrom(param.layer(i));
    // Replace any shared bottom blobs with split layer outputs.
    for (int j = 0; j < layer_param->bottom_size(); ++j) {//先改该layer的bottom的名字
      const pair<int, int>& top_idx =
          bottom_idx_to_source_top_idx[make_pair(i, j)];
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {//对每一个bottom，如果其对应的top对应多个bottom，就改该bottom的名字为 
        //【top的layer name，该bottom的blob name，top在其层中是第几个top，该bottom是top的第几个连接】
        const string& layer_name = layer_idx_to_layer_name[top_idx.first];
        const string& blob_name = layer_param->bottom(j);
        layer_param->set_bottom(j, SplitBlobName(layer_name,
            blob_name, top_idx.second, top_idx_to_bottom_split_idx[top_idx]++));
      }
    }
    // Create split layer for any top blobs used by other layer as bottom
    // blobs more than once.
    //若一个top连多个bottom，就在其上加一层，layer.type为split，layer.bottom为该top，layer.top为该top对应的多个bottom
    for (int j = 0; j < layer_param->top_size(); ++j) {//再改该layer的top的名字，注意与上面改的bottom的对应
      const pair<int, int>& top_idx = make_pair(i, j);
      const int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {//如果该top对应多个bottom
        const string& layer_name = layer_idx_to_layer_name[i];
        const string& blob_name = layer_param->top(j);
        LayerParameter* split_layer_param = param_split->add_layer();//加一个split层
        const float loss_weight = top_idx_to_loss_weight[top_idx];
        ConfigureSplitLayer(layer_name, blob_name, j, split_count,
            loss_weight, split_layer_param);
        if (loss_weight) {
          layer_param->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }
}

void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_count, const float loss_weight,
    LayerParameter* split_layer_param) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);//加的split层的bottom是之前的top名
  split_layer_param->set_name(SplitLayerName(layer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {//依次加上top层，这些top层的名字要对应原来top对应的bottom层
    split_layer_param->add_top(
        SplitBlobName(layer_name, blob_name, blob_idx, k));
    if (loss_weight) {
      if (k == 0) {
        split_layer_param->add_loss_weight(loss_weight);
      } else {
        split_layer_param->add_loss_weight(0);
      }
    }
  }
}

string SplitLayerName(const string& layer_name, const string& blob_name,
    const int blob_idx) {
  ostringstream split_layer_name;
  split_layer_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split";
  return split_layer_name.str();
}

string SplitBlobName(const string& layer_name, const string& blob_name,
    const int blob_idx, const int split_idx) {
  ostringstream split_blob_name;
  split_blob_name << blob_name << "_" << layer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return split_blob_name.str();
}

}  // namespace caffe
