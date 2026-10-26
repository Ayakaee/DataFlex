# 多模态 LEGO 示例

本目录包含一对配置：`llava15.yaml` 设置模型、数据和训练参数，`components.yaml`
设置组件及其组合顺序。默认组合为 Loss → Static → SAW → 多模态 Loss。

## 运行

在 DataFlex 根目录执行。先修改 `llava15.yaml` 中的模型路径、`dataset_dir` 和输出目录，
并在数据目录的 `dataset_info.json` 中注册训练域与 anchor。

```bash
CUDA_VISIBLE_DEVICES=0 dataflex-cli train examples/train_lora/lego/llava15.yaml
```

使用 HF 格式的 LLaVA-1.5 checkpoint；训练图文保留图片输入与回答标签。
默认训练集名称为 `coco_description,coco_short_vqa`，anchor 名称为 `anchors`。
这些名称是本地数据注册项，需要自行准备，不会自动下载。
ShareGPT 数据可按以下格式注册各数据集（anchor 同样注册）：

```json
{
  "coco_description": {
    "file_name": "coco_description.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"}
  }
}
```

每条记录包含 `conversations`（human/gpt 对话）和 `images`（图片路径列表）；
问题中的 `<image>` 数量须与图片数量一致。相对图片路径以配置的媒体根目录为准，
也可使用绝对路径。Anchor 应与训练图片分离，不使用 benchmark 测试答案。
保留 `packing: false`、`streaming: false`、`val_size: 0` 和 `mix_strategy: concat`。

## 调整组合

修改 `components.yaml` 中 `pipelines.multimodal_lego.params.stages` 的 `component`：

| 阶段 | 本次适配范围 |
| --- | --- |
| selector | `loss`、`delta_loss`、`adapt_inf` |
| mixer | `static`、`random`、`odm` |
| reorder | 示例使用 `lego_dynamic_saw`；其他已有排序模式按对应配置使用 |
| weighter | `multimodal_loss`、`multimodal_deltaloss`、`multimodal_adapt` |

将 `score_sharing` 从 `shared` 改为 `independent`，可关闭跨组件共享；更换模式需新开输出目录。
域比例的长度、顺序须与训练数据集一致。ADAPT weighter 使用 `eval_dataset` 中的 anchor。
DeltaLoss selector 衡量跨训练时刻的 loss 变化；DeltaLoss weighter 衡量无图与有图的 loss 差。
ADAPT-INF filter 保留作者 CCS 的实际选择数量，不保证恰好等于请求比例；后续阶段负责训练预算。

本例为 6 步小规模配置，刷新间隔为 2，ADAPT-INF 的 K=2。正式实验应另行对齐模型起点、
数据划分、聚类设置、学习率、训练预算及刷新周期。Loss/DeltaLoss weighter 刷新覆盖全训练池；
共享只消除兼容的重复评分。ADAPT 自身额外前向保留。ODM 原有首个 domain loss 归属逻辑未修改。

恢复训练时设置 `resume_from_checkpoint` 指向同一输出目录内的最新 checkpoint，
保持数据、图片、预处理、组件配置和共享模式一致。
