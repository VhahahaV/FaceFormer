# FaceFormer - Audio-Driven Facial Animation Baseline

FaceFormer是一个音频驱动的面部动画生成baseline模型，按照统一的训练和推理规范实现，支持多数据集训练和评估兼容的推理输出。

## 代码结构

```
FaceFormer/
├── config/                 # 配置文件
│   └── faceformer.yaml    # 主配置文件
├── dataset/               # 数据集处理
│   ├── data_item.py      # 数据项定义
│   └── dataset.py        # 数据集加载器
├── main/                  # 主要脚本
│   ├── train.py          # 训练脚本
│   ├── inference.py      # 推理脚本
│   ├── render.py         # 渲染脚本
│   └── train_eval_loop.py # 训练循环
├── models/                # 模型定义
│   ├── faceformer.py     # FaceFormer模型
│   ├── wav2vec.py        # Wav2Vec2模型
│   └── __init__.py
├── utils/                 # 工具函数
│   ├── config.py         # 配置加载
│   └── utils.py          # 通用工具
├── loss/                  # 损失函数
│   └── loss.py           # FaceFormer损失
├── outputs/              # 训练输出
├── results/              # 推理结果
├── train.py              # 主训练脚本
├── inference.py          # 主推理脚本
└── README_faceformer.md  # 本文档
```

## 支持的数据集 (按照规范配置)

### 数据集配置
- **数据根目录**: `/home/caizhuoqiang/hdd/data`
- **统一帧率**: 所有数据集重采样到25fps
- **motion维度**: 51维 (50维expr + 1维jaw_pose)

### 1. Digital Human (训练用)
- **格式**: 旧版FLAME (`expcode`, `posecode`)
- **jaw位置**: `posecode[:, 3:4]`
- **JSON**: `dataset_jsons/digital_human.json`
- **fps**: 25

### 2. MEAD_VHAP
- **格式**: 新版FLAME (`expr`, `jaw_pose`)
- **JSON**: `dataset_jsons/splits/MEAD_VHAP_train.json` (训练)
- **JSON**: `dataset_jsons/splits/MEAD_VHAP_val.json` (验证)
- **JSON**: `dataset_jsons/splits/MEAD_VHAP_test.json` (测试)
- **原始fps**: 25, **统一fps**: 25

### 3. MultiModal200
- **格式**: 类似MEAD_VHAP
- **JSON**: `dataset_jsons/splits/MultiModal200_train.json` (训练)
- **JSON**: `dataset_jsons/splits/MultiModal200_val.json` (验证)
- **JSON**: `dataset_jsons/splits/MultiModal200_test.json` (测试)
- **原始fps**: 20, **统一fps**: 25 (重采样)

## 安装和环境配置

```bash
# 创建conda环境
conda create -n faceformer python=3.10
conda activate faceformer

# 安装PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
pip install -r requirements.txt

# 安装transformers和其他依赖
pip install transformers tensorboardX tqdm pyyaml
```

## 配置说明

主配置文件位于 `config/faceformer.yaml`，包含以下主要部分：

### DATA
- `data_root`: 数据根目录
- `data_jsons`: 数据集JSON配置文件列表
- `train_subjects/val_subjects/test_subjects`: 训练/验证/测试主题

### MODEL
- `audio_encoder_repo`: 音频编码器 (facebook/wav2vec2-base-960h)
- `motion_dim`: motion系数维度 (51)
- `feature_dim`: 特征维度 (64)
- `period`: 位置编码周期 (30)

### TRAIN
- `lr`: 学习率 (0.0001)
- `epochs`: 训练轮数 (200)
- `batch_size`: 批大小 (1)
- `save_path`: 模型保存路径
- `device`: 训练设备 (cuda)

## 训练 (按照规范配置)

### 多数据集训练
```bash
# 激活环境
conda activate faceformer

# 设置数据根目录
export DATA_ROOT="/home/caizhuoqiang/hdd/data"

# 运行训练
bash train_faceformer.sh
```

**训练配置说明**:
- 数据集: digital_human (训练) + MEAD_VHAP + MultiModal200
- 帧率统一: 25fps
- Motion维度: 51维 (50 expr + 1 jaw)
- 损失: 基于FLAME顶点的MSE损失

### 训练输出
- 模型保存: `./outputs/faceformer_multi_dataset/best_model.pth`
- 日志: `./outputs/faceformer_multi_dataset/train.log`

## 推理 (评估兼容输出)

### 运行推理
```bash
# 激活环境
conda activate faceformer

# 设置数据根目录
export DATA_ROOT="/home/caizhuoqiang/hdd/data"

# 运行推理
bash inference_faceformer.sh
```

**推理输出格式** (兼容评估脚本):
```
results/metrics/faceformer_run/test/{exp_name}/{DATASET}/{STYLE_ID}_passionate/
├── 0.npy      # 第1帧顶点 (5023, 3)
├── 1.npy      # 第2帧顶点 (5023, 3)
└── ...
```

### 评估
使用项目评估脚本:
```bash
# 渲染视频并计算指标
bash scripts/eval_motion_guided.sh
```

## 渲染视频

### 渲染推理结果
```bash
python main/render.py
```

注意：渲染功能需要安装额外的依赖：
```bash
pip install trimesh pyrender
```

## 数据格式说明

### Motion系数 (51维)
- **前50维**: 表情系数 (expr)
- **第51维**: 下颌姿态 (jaw_pose的第一个维度)

### FLAME系数重建
推理时会将51维motion系数重建为完整的287维FLAME系数：
- 形状参数: 使用模板或设置为默认值
- 表情参数: 使用预测的50维expr
- 姿态参数: jaw_pose扩展为3维，头部旋转设为0
- 相机参数: 设为0
- 细节参数: 使用模板或设为0

## 模型架构

FaceFormer采用Transformer解码器架构：

1. **音频编码器**: Wav2Vec2预训练模型
2. **特征映射**: 将音频特征映射到模型维度
3. **位置编码**: 周期性位置编码
4. **Transformer解码器**: 自回归生成motion系数
5. **输出映射**: 将特征映射回motion系数空间

## 训练策略

- **教师强制**: 训练时使用真实目标作为输入
- **自回归推理**: 推理时逐步生成序列
- **梯度累积**: 支持大batch训练
- **学习率调度**: Adam优化器，默认1e-4

## 注意事项

1. **内存管理**: 大模型需要充足GPU内存
2. **数据质量**: 确保音频和FLAME数据同步
3. **格式兼容**: 不同数据集的FLAME格式需要正确处理
4. **序列长度**: 自动截断到300帧以适应内存

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 启用梯度累积
   - 截断序列长度

2. **数据加载失败**
   - 检查JSON配置文件路径
   - 验证数据文件存在
   - 检查FLAME数据格式

3. **训练不收敛**
   - 检查学习率设置
   - 验证数据预处理
   - 调整模型参数

### 日志和调试

训练日志保存在 `outputs/*/train.log`
推理日志保存在 `results/inference.log`
