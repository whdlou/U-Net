# U-Net 2D语义分割网络

## 文件结构

```
U-Net
├─ cfg.py // 配置文件
├─ data.py // 自定义数据读取
├─ dataset  // 数据集根目录
│    ├─ test // 测试集
│    │    ├─ images // 测试集图片
│    │    └─ masks // 测试集标签（如果未提供标签可以为空）
│    ├─ train // 训练集
│    │    ├─ images // 训练集图片
│    │    └─ masks // 训练集标签
├─ inference.py // 推理
├─ main.py // 主函数，执行训练，推理代码
├─ models // 模型文件夹
│    └─ unet.py // U-Net网络模型
├─ output
│    ├─ losses // 存放训练期间损失函数文件
│    │    └─ README.md
│    ├─ results // 存放推理生成的结果掩码图
│    │    └─ README.md
│    └─ weights // 存放训练期间的权重文件
│           └─ README.md
├─ plot.py // 绘制损失函数图
├─ requirements.txt // 环境
├─ tools.py // 工具
└─ train.py // 训练代码
```

## 使用说明

1. 训练/测试数据集放在对应文件夹下，删除`train.txt`, `val.txt`, `test.txt`

2. 调用`data.py`中的`generate_data_file()`生成新的数据集列表文件，传入rate参数可以修改验证集比例，例如`generate_data_file(rate=0.2)`，默认`rate=0.3`

3. 修改配置文件

   ```
   img_type = '.jpg' # 输入图片的格式
   mask_type = '.png' # 输入label的格式
   palette = ((0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255)) # 调色板，除了第一个背景颜色之外其它颜色可以随意更改，支持三通道或单通道（单通道第一个元素改成0）
   num_classes = 4 
   size = (256, 256)
   batch_size = 16
   start_epoch = 0
   num_epochs = 250
   lr = 1e-3
   weight = 'output/weights/epoch171_loss0.012811.pth' # 选择的权重位置
   ```

4. 运行`main.py`，训练期间注释掉训练后面的代码，推理期间注释掉训练代码
