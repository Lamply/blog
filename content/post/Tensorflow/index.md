---
title: TensorFlow
date: 2020-05-29
image: 
description: 
math: 
categories:
  - 技术简介
  - 问题记录
tags:
---
## TensorFlow 2.x Guild
### 2.0 changes
RFC: https://github.com/tensorflow/community/pull/20  
默认启用 eager execution（eager mode），原本只能通过 tf 函数建静态图，用 session 作为来包装 tensorflow 的状态（graph mode）。现在通过 `tf.function()` 装饰 python 函数也可以使 tensorflow 运行等价的图（ AutoGraph 实现 ），利于优化保存。  
将 1.x 版本代码转为 2.x 代码：`tf_upgrade_v2`

### Recommendations
- 将代码写成更小的函数，high-level 的计算通过 `tf.function` 来装饰
- 多用 keras
- 从硬盘读取数据流使用 `tf.data.Dataset`
- 使用 python 控制流来获得 AutoGraph 便利（ 自动转换成相应图模式的等效形式 ）
- 使用 `tf.metrics` 来聚合数据， `tf.summary` 来 log

Check this for more details: https://github.com/tensorflow/docs/blob/master/site/en/guide/effective_tf2.md
## 结构
### 数据格式
Tensorflow：  
Tensor：[N, H, W, C]  
conv kernel：[h, w, ci, co]  
dense kernel：[ci, co]

Caffe：  
Tensor：[N, C, H, W]  
conv kernel：[co, ci, h, w]

Chainer：  
Tensor：[N, C, H, W]  
conv kernel：[co, ci, h, w]

Pytorch：  
Tensor：[N, C, H, W]  
conv kernel：[co, ci, h, w]

### tf.function
使代码以图形式独立出来，加快运行速度和效率，需要注意几点：  
- 在 Eager 模式 debug，完事后用 `@tf.function` 来装饰函数
- 不要依赖 python 的特征如对象变动、列表插入
- 最好用 TensorFlow 的自带算子

在使用时，内部函数只能使用 tf 自带函数或部分 python 特性的程序，不同类型的 Tensor 操作需要先转换类型  

参考：  
https://tensorflow.google.cn/guide/function  
https://zhuanlan.zhihu.com/p/67192636  
https://zhuanlan.zhihu.com/p/127189133
      

### tf.data.Dataset
- `.repeat(count=None)`：数据集增加重复 `count` 次，`None` 或 `-1` 则一直重复，__注意数据格式要求为 `tf.int64`，不要设成 `True` 或 `False`，否则可能会炸掉拿不到数据（还不会报错警告）__
- `.map(map_func)`：对数据集内每一个元素采用 map_func 做映射。__注意！__这里的元素内部是 Tensor 类型的（维度为 3，HWC），而非默认的 EagerTensor 类型，也就是说不能使用任意的 python 函数，不能转成 numpy 处理，应该绕过 numpy 用 Tensor 的方法进行处理，若非要用自定义函数则参照官方文档描述：
  > Note that irrespective of the context in which map_func is defined (eager vs. graph), tf.data traces the function and executes it as a graph. To use Python code inside of the function you have two options:
  > 
  > 1) Rely on AutoGraph to convert Python code into an equivalent graph computation. The downside of this approach is that AutoGraph can convert some but not all Python code.
  > 
  > 2) Use tf.py_function, which allows you to write arbitrary Python code but will generally result in worse performance than 1). For example:
- `.shuffle(buffer_size)`：
- `.batch(batch_size, drop_remainder=False)`：设置批大小，`drop_remainder` 参数表明当数据集剩下不足以凑一个 batch 时是否去掉剩余的部分，默认不去掉（会导致 epoch 最后一个 batch 大小可能变小）

### tf.GradientTape
在 Custom Training 时用到，记录观察变量的计算过程（默认只记录 `tf.Variables`），并可以对观察变量做梯度求导
  - `persistent`：设为 `True` 则可以多次调用 `.gradient()`
  - `.gradient(loss, model.trainable_weights)`：计算得到相关 weights 的梯度，list 结构，元素为和 weights 相同 shape 的 Tensor

### tf.distribute.Strategy
- 同步/异步训练：同步训练通过 all-reduce 实现，workers 同时训练不同的输入数据，每个 step 后将梯度 aggregating 起来。异步训练通过 parameter server 架构实现，workers 独立训练输入数据，异步更新变量。
- 硬件平台：可以 scale 到单机多卡或多机多卡或者云 TPU 上。
- 为此提供了六种策略（v2.2）：

|     Training API     | Mirrored Strategy | TPU Strategy  | MultiWorker Mirrored Strategy | Central Storage Strategy | ParameterServerStrategy    | One Device Strategy |
| :------------------: | :---------------: | :-----------: | ----------------------------- | ------------------------ | -------------------------- | ------------------- |
|      Keras API       |     Supported     |   Supported   | Experimental support          | Experimental support     | Supported planned post 2.3 | Supported           |
| Custom training loop |     Supported     |   Supported   | Experimental support          | Experimental support     | Supported planned post 2.3 | Supported           |
|    Estimator API     |  Limited Support  | Not supported | Limited Support               | Limited Support          | Limited Support            | Limited Support     |

主要用到的两种：
- MirroredStrategy：支持单机多卡同步训练，它为每个 GPU 创建副本，模型的变量是所有副本的镜像。这些变量构成 `MirroredVariable` 概念，通过同一更新来保持同步。变量在设备间通过 all-reduce 算法来通信，可以明显减小同步开销，默认使用 NVIDIA NCCL 来实现。
- MultiWorkerMirroredStrategy：类似 MirroredStrategy，在多机多卡上同步训练，使用 CollectiveOps 作为多机 all-reduce 通信方法。collective op 是一个根据运行时硬件、网络拓扑和张量大小自动选择 all-reduce 算法 Tensorflow 图的算子。同时也做额外的性能优化。有两种 collective ops 实现方法，ring-base 使用 gRPC 作为通信，还有就是 nccl。多机多卡训练和单机多卡训练最关键的不同点在于多机的设置，Tensorflow 标准方法是使用 TF_CONFIG 环境变量来设置。

### tf.assign 和 tf.identity
用于给变量赋值或读取同步（由于使用计算图时只计算更新节点相关的部分变量，会导致某些不参与计算的节点参数不会更新，比如复制 `=` 的输出节点，当然这个仅限于 v1 版本，v2 版本下 Eager 和 Autograph 都不需要考虑这个问题）。  

在分布式训练中，`tf.assign` 无法对 weights 使用，因为 `keras.layers` 的初始化参数没有指定 `aggregation`，更新值无法聚合（虽然通过修改源码应该可以做到），通过直接赋值修改引用应该可以使得一些不用更新的操作在分布式训练中变得可行。  

#### tf.control_dependencies
使用计算图模式时因为是喂数据进某些节点然后计算某些输出，其中输出间的前后关系有时是混淆的，不一定会像期望那样顺序执行，所以需要用这个来设定必须在执行了某个节点计算之后才做的一些操作。  

在 v1 版本中，该函数经常配合 `tf.identity` 来将不在计算图中的输出节点连接到计算图中用以在计算时同步输出值，v2 版本的 Eager 和 Autograph 都不会用到，因为此时代码会按期望的运行，只有在 `Dataset.map` 这种 graph context 下才会用到。



### Tensor
- 读取二进制消息 Tensor：`tf.io.decode_raw(data, 'uint8')`

### TFRecord
序列化数据，可高效读取。通过 protobuf 实现？对应于 Tensorflow 中使用的是 `tf.Example` 消息。通过 `tf.Example` 来创建或读取 `.tfrecord`。  
`tf.Example` 是 `{"string": tf.train.Feature}`  的数据映射，支持三种基本类型 `tf.train.BytesList`、`tf.train.FloatList`、`tf.train.Int64List`，将 tensorflow 类型转换为这三种类型来构造 `tf.Example` 并序列化得到 `.tfrecord`。

更多内容见：https://www.tensorflow.org/tutorials/load_data/tfrecord

### Model
注意，Tensorflow 的计算图可以视作程序，可以做任何事，在安全性方面上有很多需要考量的问题：https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md

- Checkpoint：包含模型参数和训练状态，其中
  - 有一个或多个包含模型参数的碎片，单机下后缀为：`.data-00000-of-00001`
  - 表明哪些参数在哪些碎片里面的索引文件
- SavedModel / HDF5：包含参数、模型架构、训练配置、优化器和状态等一切
  - HDF5 使用对象配置保存模型架构，而 SavedModel 以执行图来保存。所以 SavedModel 可以保存自定义对象  
- JSON + HDF5：用于本地推理的结构，和 SavedModel 比起来不会保存训练的信息，模型读取速度也会快非常多，但是需要 import 模型的自定义层，否则从 json 解析不了自定义的 layer
  ```python
  # 写
  model.trainable = False
  model.save_weights('weights.h5')
  model_json = model.to_json()
  with open('model.json', "w") as json_file:
      json_file.write(model_json)
      
  # 读
  with open("model.json", 'r') as json_file:
      loaded_model_json = json_file.read()
model = tf.keras.models.model_from_json(loaded_model_json)
  model.load_weights("weights.h5")
  model.trainable = False
  ```

模型的分析更多见：https://zhuanlan.zhihu.com/p/159759699

### Profiler
用于分析训练性能的工具，需要用到 CUPTI，通过捕获一些 step 的各个算子计算时间来分析模型训练性能。  
发现的一个事实是： Batch Norm 的 FusedBatchNormV3 其实挺耗时间的，而且需要在 Host 做，不过换成其他 Normalization 方法其实也差不多。如果模型计算量本身比较小的话，那 GPU 利用率会降低不少。

## 方法
#### 计算
- `tf.reduce_sum()`：类似 `np.sum()`，区别在于它的输出保存在__相同的类型__的 Tensor 中，也就是说输入 `[254, 44, 44], dtype=uint8`，返回的值为 `86`，类型是 `uint8`，不会有溢出警告
- `tf.gather(data, idx)`：从 data 中取出 idx 号元素（整型 Tensor）形成一个 Tensor，也就是类似切片

#### 随机
为了让每次实验可重复，设置好随机是必须的，而设置确定的随机需要几个步骤：
1. 给参数初始化、数据集 on-the-fly 扩增等依赖随机的部分设置种子，并自动记录下来
2. 给 numpy、random 库设置全局种子
3. 在训练命令之前 `export TF_CUDNN_DETERMINISTIC=true` 可以让 cudnn 使用确定性算法（seed == 0）

#### 类型转换
```python
t2 = tf.cast(t1, dtype=tf.float32)
```

#### 模型输入
设置 `input_shape=(None, None, 3)` 可以允许动态输入，可以在输出推理模型时指定

#### 数据加载
```python
## From numpy array, the first axis should be the same?
  # x_train: (60000, 28, 28, 1)
  # y_train: (60000,)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(32)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

for images, labels in train_data:   # iterables
	pass

## From CSV
column_to_use = [0, 1, 2, 3, 4, 8]
record_defaults = [tf.int32, tf.int32, tf.string, tf.string, tf.float32, tf.float32]
data = tf.data.experimental.CsvDataset("titanic_dataset.csv", record_defaults, header=True, select_cols=column_to_use)



## From TFRecord
dataset = tf.data.TFRecordDataset(path)

  # Create a dictionary describing the features.
feature_desc = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'mask_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_desc)

train = dataset.map(_parse_image_function)
# train['image_raw']: Tensor
```
#### 模型加载保存
```python
## Save/Load checkpoint
# Save in model fitting callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1)
# Manually save
model.save_weights(checkpoint_path.format(epoch=0))

model.load_weights(checkpoint_path)

## Save/Load entire model (resume training or ...)
# SavedModel format (default in TF2)
model.save('my_model') 

# keras h5 format
model.save('my_model.h5') 

tf.keras.models.load_model(xxx)    # compile with same arguments

model = create_model()
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)   # load checkpoint
```

#### 模型训练
```python
model = tf.keras.Model(inputs=x, outputs=y)

## 1 Use Keras API to train
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_dataset)


## 2 Use tf.GradientTape
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
```

#### 模型转换
```python
## Convert to TFLite
# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model('efficientb0_UNet')
tflite_model = converter.convert()
# Save the TF Lite model.
with tf.io.gfile.GFile('efficientb0_UNet.tflite', 'wb') as f:
    f.write(tflite_model)
    
    
    
## Convert to Frozen Graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)
```
详细资料：https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/

#### 模型修改
keras 模型似乎会被保护，无法直接通过赋值等手段更改，似乎只能重新构建一个模型？测试通过 `set_weights([filters, bias])` 可以修改。
```python
psmodel = unet_model(2)

# For nested model
model.layers[1].save_weights('weights/efficientb0_UNet_81_f1score/efficient_sgd_81_30e_24bs_20200616_backbone.h5')
psmodel.layers[1].load_weights('weights/efficientb0_UNet_81_f1score/efficient_sgd_81_30e_24bs_20200616_backbone.h5')

# For each layer
for i in range(2, len(psmodel.layers)-1):
    if(type(psmodel.layers[i]) == tf.keras.Sequential):
        psmodel.layers[i].set_weights(model.layers[i].get_weights())

filter_81 = model.layers[-1].get_weights()[0][...,:2,:]
bias_81 = model.layers[-1].get_weights()[1][...,:2]
psmodel.layers[-1].set_weights([filter_81, bias_81])

psmodel.save('efficientb0_UNet_f1score/efficient_sgd_81to2_20200616-183119')
```

#### 自定义 loss / metrics
```python
## 以 metrics 方式，y_true 和 y_pred 都是 [batch_size, height, width, channel] 的 graph mode tensor
def f1score(y_true, y_pred):
    y_pred_ = tf.argmax(y_pred, axis=-1, output_type=tf.int64)[...,tf.newaxis]
    y_true_ = tf.cast(y_true, dtype=tf.int64)
    
    tmp_1 = tf.reduce_sum(tf.multiply(y_true_, y_pred_), axis=[1,2,3])
    tmp_2 = tf.reduce_sum(y_pred_, axis=[1,2,3])
    tmp_3 = tf.reduce_sum(y_true_, axis=[1,2,3])
    ttmp_1 = tf.cast(tmp_1, dtype=tf.float32)
    ttmp_2 = tf.cast(tmp_2, dtype=tf.float32)
    ttmp_3 = tf.cast(tmp_3, dtype=tf.float32)
    
    precision = ttmp_1 / (ttmp_2+0.000001)
    recall = ttmp_1 / (ttmp_3+0.000001)
    f1_score = 2. * precision * recall / (precision + recall + 0.000001)
    return f1_score

model.compile(..., loss=xxx, metrics=[f1score])

  # 读取时需要添加自定义对象
model = tf.keras.models.load_model('xxx', custom_objects={"f1score":f1score})

## 以 callback 方式
```

#### 迁移学习
```python
## Rough
base_model = ...
base_model.trainable = False

# function api
def new_model():
	inputs = tf.keras.layers.Input(shape=[224, 224, 3])
	x = base_model(inputs)
	y = model_head(x)
    return tf.keras.Model(inputs=inputs, outputs=y)
model = new_model()

# or in sequential api
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])
```

#### 性能分析
使用 TensorFlow Profiler：`pip install -U tensorboard_plugin_profile
`    
抓取一些 step 来分析性能，输出保存到 path
```python
if step == 120:
	tf.profiler.experimental.start(path)
if step == 180:
    tf.profiler.experimental.stop()
```

#### 数据/模型分析
What If Tool

##### 参数显著性分析
```python
insignificant = []
for i in range(len(model.layers)):
    for j in range(len(model.layers[i].weights)):
        if "kernel" in model.layers[i].weights[j].name:
            sum_k = np.mean(np.abs(model.layers[i].weights[j].numpy()), axis=(0,1,2))
            norm_sum_k = sum_k / sum_k.max()
            for v in range(norm_sum_k.shape[0]):
                if norm_sum_k[v] < 1e-3:
                    insignificant += [(i, j, sum_k[v], sum_k.max(), model.layers[i].weights[j].name)]
                    print(i, j, sum_k[v], sum_k.max(), model.layers[i].weights[j].name)
```

#### 分布式训练
使用 `tf.keras` 来实现：  
```python
# 单机多卡
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = xxx
  model.compile(loss='mse', optimizer='sgd')

model.fit(dataset)

num_workers = 4


# 多机多卡
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset)
```

使用 `tf.GradientTape` 来实现单机多卡： 
```python
strategy = tf.distribute.MirroredStrategy()
global_batch_size = per_worker_batch_size * strategy.num_replicas_in_sync

## 定义模型和优化器
with mirrored_strategy.scope():
    G = get_generator([flags.z_dim])
    D = get_discriminator([flags.output_size, flags.output_size, flags.c_dim])

    G.trainable=True
    D.trainable=True

    d_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    g_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    
## 分发数据集
with mirrored_strategy.scope():
    images, images_path = get_anime(flags.output_size, flags.n_epoch, GLOBAL_BATCH_SIZE)
    dist_images = mirrored_strategy.experimental_distribute_dataset(images)

## 定义损失函数，注意 reduce 需要用 global batch size
with mirrored_strategy.scope():
    def compute_loss(fake_logits, real_logits):
        real_d_l = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_logits), real_logits)
        fake_d_l = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_logits), fake_logits)

        d_loss = fake_d_l + real_d_l
        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_logits), fake_logits)

        # 将 loss 各个元素视为一个 example，累加起来除以 GLOBAL_BATCH_SIZE
        return tf.nn.compute_average_loss(d_loss, global_batch_size=GLOBAL_BATCH_SIZE), \
               tf.nn.compute_average_loss(g_loss, global_batch_size=GLOBAL_BATCH_SIZE)
               
## 定义训练函数
def step_fn(batch_images):
    with tf.GradientTape(persistent=True) as tape:
        z = tf.random.normal([flags.batch_size, flags.z_dim], mean=0.0, stddev=1.0)
        fake_images = G(z, training=True)

        fake_logits = D(fake_images, training=True)
        real_logits = D(batch_images, training=True)

        d_loss, g_loss = compute_loss(fake_logits, real_logits)

    g_grad = tape.gradient(g_loss, G.trainable_weights)
    g_optimizer.apply_gradients(zip(g_grad, G.trainable_weights))
    d_grad = tape.gradient(d_loss, D.trainable_weights)
    d_optimizer.apply_gradients(zip(d_grad, D.trainable_weights))
    del tape
    return d_loss, g_loss

@tf.function
def distributed_train_step(dataset_inputs):
    replica_d_losses, replica_g_losses = mirrored_strategy.run(step_fn, args=(dataset_inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, replica_d_losses, axis=None), \
           mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, replica_g_losses, axis=None)

## 训练
for epoch in range(flags.n_epoch):
    sum_d_loss = 0.0
    sum_g_loss = 0.0
    num_step = 0
    step_time = time.time()

    # 分发策略采用数据分发，每张卡有同等大小的 batch size
    for x in dist_images:
        # 去掉每个 epoch 最后一个不完全的 batch
        if x.values[0].shape[0] != BATCH_SIZE_PER_REPLICA:
            break
        tmp_d_loss, tmp_g_loss = distributed_train_step(x)
        sum_d_loss += tmp_d_loss
        sum_g_loss += tmp_g_loss
        num_step += 1

    d_loss = sum_d_loss / num_step
    g_loss = sum_g_loss / num_step

```

## 问题
- `Tensor.name is meaningless when eager execution is enabled`
  - TF1 转 TF2 问题，能升级升级，不能升级就算了
- `Cannot batch tensors with different shapes in component 0. First element had shape [224,224,3] and element 3 had shape [224,224,1]`：部分输入数据存在问题，比这个是存在灰度图，channel 和 rgb 图不一致，导致无法 batch 起来
- 使用 keras 训练的模型，转换成 TFLite 后使用的算子和转成 Frozen Graph 后使用的算子有所不同（如 Transpose Conv）
- `Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.`
  - 原因不明，当报出 `dlerror: libcupti.so.10.1: cannot open shared object file: No such file or directory;` 时有出现，无法重现
  - `export TF_FORCE_GPU_ALLOW_GROWTH=true` 似乎可以解决
  - 似乎又不能解决，总之和显卡有非常大关系
- `Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR`：随上面的错误发生，似乎是显存分配之类的问题，尝试 `export TF_FORCE_GPU_ALLOW_GROWTH=true` 使 TensorFlow 按需分配显存
- `Segmentation fault`
  - 报在加载 `libcupti.so.10.1` 之后，原因大概是因为使用 `libcupti.so.10.2` 来创建符号链接 `libcupti.so.10.1`（Tensorflow2.2 不能正常支持 CUDA10.2），TensorBoard callback 需要用到 cupti（但似乎不用也可）
- Keras 的 `tf.keras.metrics.Accuracy()` 和模型 compile 时用的 Accuracy 算出来的结果不同：
  - compile 时使用的是 built-in function，可能有区别？
  - 无法重现......
- Keras sparse class crossentropy 在官方文档上看到的计算结果好像和手算的不一样
  - 在 tensorflow 后端下，`crossentropy` 的实现为：先对预测值 scale 到 [0, 1]，然后 clip 到 [1e-7, 1-1e-7]（防止 log 零值），最后 log 预测值（自然对数），取出正确类别对应的 log 值，对所有样本求平均取负数。也就是结果为： -logα/n
  - 应当是手算的理解有所出错，多分类和二分类计算概念混淆
  - 要注意这里 Tensorflow 版的输入顺序是 (label,logits)，和 Pytorch 版相反
- `tf.ResizeNearestNeighbor' op is neither a custom op nor a flex op`：
  ```
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                         tf.lite.OpsSet.SELECT_TF_OPS]
  ```
- 想在 `.map()` 中检查输入 Tensor 的维度 `image.shape[-1]`，但是完全不行，即使进了判断语句执行了 gray 转 rgb 出来的仍然是 gray 图，完全不明所以，只能通过外部解决然后再作为参数输入
  - 发现不能用 `image.shape[-1]` 来获得 shape，而应该使用 `tf.shape(image)[-1]`
- keras 同一上下文下好像层的默认名字会随创建次数改变，也就是说再创建同一个模型，其默认层名的后缀数字会顺序增加，惊了
- `model.load_weights()`：它的参数 `by_name` 和 `skip_mismatch` 只支持读取 h5 文件时使用，futher over，嵌套模型在此也存在问题，最好还是不要在模型中嵌入模型了
  > Only topological loading (`by_name=False`) is supported when loading weights
from the TensorFlow format. Note that topological loading differs slightly
between TensorFlow and HDF5 formats for user-defined classes inheriting from
`tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
TensorFlow format loads based on the object-local names of attributes to
which layers are assigned in the `Model`'s constructor.
- `Nan in summary histogram for: sequential/conv2d/kernel_0 [Op:WriteHistogramSummary]`：
  - 可能是 loss 出了问题，应该检查下输出层结构和 loss func，把 tensorboard 关掉再训练可以看到 loss 如何
  - 也可能是训练数据存在问题
  - 也可能是 from scratch 训练初期不稳定导致
- `cuDNN launch failure : input shape ([64,4,4,512]) [Op:FusedBatchNormV3]`
  - 很可能是显卡等原因，换成 V100 远程机没有出现这个问题
  - 原因大概是显存容量不足或被占用的问题
  - 通过设置环境变量 `export TF_FORCE_GPU_ALLOW_GROWTH=true` 以及低的 batch size 也许能解决这个问题
- `ValueError: You must specify an aggregation method to update a MirroredVariable in Replica Context. You can do so by passing an explicit value for argument`
  - 大体意思是要给 `tf.Variable()` 指定 `aggregation=tf.VariableAggregation.SUM`，否则不知道怎么聚合。在多 GPU 训练时出现
  - 应该就是字面意思，哪里用到了 `tf.Variable()` 没指定 `aggregation` 参数，比如 optimizer 的 lr_scheduler
  

- `Internal: failed call to cuDevicePrimaryCtxRetain: CUDA_ERROR_INVALID_DEVICE: invalid device ordinal`：
  - 可能是其他 TF 程序（如 notebook）占用了显卡，`nvidia-smi` 看看关掉占用进程试试
- `TypeError: Failed to convert object of type <class 'list'> to Tensor. Contents: [None, 4, 4, 1]. Consider casting elements to a supported type.`：
  - 取 Tensor 维度时用了 `xxx.shape`，同时拿到的 batch_size 维度是 None
  - `xxx.shape` 或 `xxx.get_shape` 返回的是 `tf.TensorShape` 类型，不属于 `Tensor`
  - 再用于构建 list 时无法用 None 构建 Tensor，所以报错。应该使用 `tf.shape(xxx)` 来获取含 None 的维度 Tensor
- TFRecord 读取错误 `InvalidArgumentError: Key: xxx.  Can't parse serialized Example.`：
  - 保存数组时 fixed length feature 的 description 要指定读出数组的大小，如 `'landmarks': tf.io.FixedLenFeature([212], tf.float32),`
- CUDA 11.1 出现问题 `libcusolver.so.10 not found`
  - `ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11 $(python -c "import tensorflow.python as x; print(x.__path__[0])")/libcusolver.so.10`
- 模型读取速度极其慢（需要数分钟）
  - 和模型存储的硬盘的 I/O 速度有关，使用 `tf.keras.models.load_model` 读取模型可能会大量用到 I/O ，所以如果是在服务器的话最好就先把模型移到本地再读取
  - 同时很可能和保存格式有关， `savedmodel` 的保存格式会极其慢，最好还是分开 weights 和模型结构保存，或者直接构建网络再读取 weights
  - 也有可能跟 tensorflow 版本有关，已知 TF2.2 带自定义层的模型无法在 TF2.3 以上版本加载，会无限卡住



-------------------------------
## 1.0 (deprecate)
### 简介
1.0 版没有 eager mode，每个节点都得事先定义，定义完计算图后再计算。

静态图变量通过变量作用域共享，使用命名空间可以随地拿到变量
- `tf.get_variable()`：创建变量或获取变量
- `tf.variable_scope()`：命名空间上下文
  - `reuse=False`：空间里的变量是新变量，否则重用空间里的变量

### 训练流程
1. `sess = tf.InteractiveSession()`创建会话，就是下面的运算符节点都会在该会话下创建。
2. 定义要用到的placeholder，即只定义计算图要用到的数据结构和shape，如：
```python
image_holder = tf.placeholder(tf.float32, [b, h, w, c])
label_holder = tf.placeholder(tf.int32, [batch_size])
keepprob_holder = tf.placeholder(tf.float32)
```
3. 然后定义计算图的前向推理(Inference)流程，也就是叠层，输入是placeholder。
```python
y_conv=inference_op(image_holder, keepprob_holder)
```
4. 接着定义根据前向输出与label计算loss的方法。
```python
loss = loss(y_conv,label_holder)
```
5. 定义loss的优化器，得到可以执行`.run()`的。
```pyhthon
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
```
6. 运行初始化所有变量的操作
```python
tf.global_variables_initializer().run()
```
7. 开始训练
```python
for i in range(20000):
    # batch = mnist.train.next_batch(50)
    image_batch, label_batch = sess.run([images_train, labels_train])
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={image_holder: image_batch, label_holder: label_batch, keepprob_holder: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={image_holder: image_batch, label_holder: label_batch, keepprob_holder: 0.5})
```
optional: 定义计算batch准确率的方法(accuracy)。

### Freeze
freeze 之前最好将模型类里不必要的东西（训练部分，绑定 sess，CycleGAN 的 BtoA 之类的）统统去掉
```python
from tensorflow.python.tools import freeze_graph 

gan = UGATIT(args)
gan.build_model()

# 可以写在 build_model() 里面
gan.test_domain_A = tf.placeholder(tf.float32, [1, gan.img_size, gan.img_size, gan.img_ch], name='test_domain_A')
gan.test_fake_B, _ = gan.generate_a2b(gan.test_domain_A)

# 指定 checkpoint
checkpoint_dir = os.path.join(gan.checkpoint_dir, gan.model_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
model_checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './models', 'MLP_deconv_avg.pb') 
    freeze_graph.freeze_graph(
        input_graph='./models/MLP_deconv_avg.pb',
        input_saver='',
        input_binary=False, 
        input_checkpoint=model_checkpoint_path, 
        output_node_names='generator_B/out',   # 输出的节点需要定义 name，最后 scope + name 为此处的值
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./models/MLP_deconv_avg_frozen_model.pb',
        clear_devices=True,  # 是否清除训练时节点指定的设备
        initializer_nodes=''
        )
```

### Notice
- 给需要保存或可视化观察的数据加`name`，用于提取时指定
- 所有变量都必须初始化，使用`tf.global_variables_initializer().run()`来执行初始化操作
    - notebook 里重新读计算图似乎要重启内核才行，不然会出现这个：`ValueError: Variable conv1/weights already exists`，或者通过 `tf.reset_default_graph()` 来重置计算图




