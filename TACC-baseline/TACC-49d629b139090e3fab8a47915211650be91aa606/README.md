# 2026 腾讯广告算法大赛 Baseline 快速上手

这份 baseline 是一个面向 PCVR 任务的 PyTorch 训练工程。主流程是：

```text
官方 Parquet 数据 + schema.json
        |
        v
dataset.py 读取多列特征、裁剪/补齐序列、生成 batch 张量
        |
        v
model.py 中的 PCVRHyFormer 编码 user/item/sequence 特征
        |
        v
trainer.py 训练、验证 AUC/logloss、保存 best checkpoint
```

你给的 `0000.json` 到 `0004.json` 是从官方 `sample_data.parquet` 读出来的单行样例，适合理解字段结构；baseline 实际训练时不是读这些 JSON，而是直接读官方提供的 `*.parquet` 多列文件。

## 与 HyFormer 论文的关系

这里的 `HyFormer` 基本指向论文《HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction》（ByteDance，arXiv:2601.12681v2）。baseline 里的实现是比赛版复现/改写，不一定包含论文工业系统里的全部训练和部署优化，但核心建模思路是一致的。

论文要解决的问题是：工业 CTR/推荐模型通常先做长序列建模，再把压缩后的序列 token 和非序列特征做 late-fusion feature interaction。这个“两阶段、单向、晚融合”的范式会限制序列信息和异构特征之间的深层交互。HyFormer 的核心想法是把长序列建模和非序列特征交互放进一个统一 backbone，让两类信息在多层里反复互相更新。

论文中有两个核心模块，对应到本 baseline 的代码如下：

| 论文概念 | 论文含义 | baseline 对应 |
| --- | --- | --- |
| Non-Sequential Tokens / Global Tokens | user、item、context 等非序列特征 token，作为长序列解码的全局语义接口 | `GroupNSTokenizer`、`RankMixerNSTokenizer`、`user_dense_proj` |
| Query Generation | 用非序列 token 和序列 mean pooling 信息生成每个序列的 query tokens | `MultiSeqQueryGenerator` |
| Query Decoding | query tokens 对长序列 K/V 做 cross-attention，从序列里取 target-aware 信息 | `CrossAttention` 和 `MultiSeqHyFormerBlock` 中的 per-sequence decoding |
| Query Boosting | 对 decoded queries 和 NS tokens 做 MLP-Mixer/RankMixer 风格 token mixing | `RankMixerBlock` |
| Multi-Sequence Modeling | 多个行为序列不要简单合并，而是每个序列独立解码，再通过 query-level mixing 交互 | `seq_domains`、每个 domain 独立 encoder/cross-attention |
| Capacity-Efficiency Trade-off | 序列编码可在 full Transformer、LONGER-style、轻量 FFN 间取舍 | `--seq_encoder_type transformer/longer/swiglu` |

论文里的一个关键公式是：

```text
Global Info = Concat(F1, ..., FM, MeanPool(Seq))
Q = FFN_1(Global Info), ..., FFN_N(Global Info)
```

baseline 的 `MultiSeqQueryGenerator.forward()` 就是在做这件事：先把 NS tokens flatten，再对每个序列做 mean pooling，然后拼成 `global_info`，通过多个 FFN 生成每个序列域自己的 query tokens。

另一个关键结构是每层都交替执行：

```text
Query Decoding:  Q_hat = CrossAttn(Q, K_seq, V_seq)
Query Boosting:  Q_next, NS_next = RankMixer/MLP-Mixer(Concat(Q_hat, NS))
```

baseline 的 `MultiSeqHyFormerBlock.forward()` 也按这个顺序走：先独立编码每个序列，再让 query 对序列做 cross-attention，最后把所有 decoded queries 和 NS tokens 拼起来交给 `RankMixerBlock`。

## 目录结构

```text
.
├── train.py        # 训练入口：解析参数、创建 DataLoader、组装模型和 Trainer
├── dataset.py      # Parquet IterableDataset：按 schema.json 把特征转成训练张量
├── model.py        # PCVRHyFormer 模型主体：NS tokenizer、多序列编码、HyFormer blocks
├── trainer.py      # 训练/验证循环：BCE/Focal loss、AUC/logloss、checkpoint
├── utils.py        # 日志、随机种子、EarlyStopping、Focal Loss
├── ns_groups.json  # GroupNSTokenizer 的示例分组配置
└── run.sh          # 官方风格启动脚本，默认启用 rankmixer NS tokenizer
```

## 数据格式

训练目录需要长这样：

```text
data_dir/
├── schema.json
├── sample_data.parquet
└── other_data.parquet
```

`schema.json` 是 baseline 的关键元信息，`dataset.py` 期望它包含：

```json
{
  "user_int": [[fid, vocab_size, dim]],
  "item_int": [[fid, vocab_size, dim]],
  "user_dense": [[fid, dim]],
  "seq": {
    "seq_a": {
      "prefix": "parquet_column_prefix",
      "ts_fid": 28,
      "features": [[fid, vocab_size]]
    }
  }
}
```

DataLoader 输出给模型的 batch 主要包括：

```text
user_int_feats       LongTensor [B, user_int_dim]
item_int_feats       LongTensor [B, item_int_dim]
user_dense_feats     FloatTensor [B, user_dense_dim]
item_dense_feats     FloatTensor [B, 0]，当前 baseline 基本为空
{seq_domain}         LongTensor [B, num_sideinfo_features, max_len]
{seq_domain}_len     LongTensor [B]
{seq_domain}_time_bucket LongTensor [B, max_len]
label                LongTensor [B]，由 label_type == 2 映射得到
```

单行 JSON 样例里可以看到同一套逻辑的嵌套表示：

- 顶层字段：`user_id`、`item_id`、`timestamp`、`label`、`user_feature`、`item_feature`、`seq_feature`
- 非序列特征：按 `feature_id` 和 `feature_value_type` 表示，类型包括 `int_value`、`int_array`、`float_array`、`int_array_and_float_array`
- 序列特征：`action_seq`、`content_seq`、`item_seq` 中每个 `feature_id` 对应一条等长数组
- 我看过的 5 条样例中，序列长度差异很大，例如 `0000.json` 的 `action/content/item` 长度分别是 `228/841/181`，`0002.json` 分别是 `3873/1177/470`，所以 baseline 会用 `--seq_max_lens` 做截断

## 快速运行

先安装依赖：

```bash
pip install torch numpy pyarrow tqdm scikit-learn tensorboard
```

推荐用 `run.sh` 启动。注意：`train.py` 会强制创建 checkpoint、log 和 TensorBoard 目录，所以以下 4 个环境变量都建议设置。

```bash
export TRAIN_DATA_PATH=/path/to/official/data_dir
export TRAIN_CKPT_PATH=/path/to/output/ckpt
export TRAIN_LOG_PATH=/path/to/output/log
export TRAIN_TF_EVENTS_PATH=/path/to/output/tf_events

bash run.sh --num_epochs 3 --batch_size 128
```

Windows PowerShell 可以直接跑 `train.py`：

```powershell
$env:TRAIN_DATA_PATH = "C:\path\to\official\data_dir"
$env:TRAIN_CKPT_PATH = "C:\path\to\output\ckpt"
$env:TRAIN_LOG_PATH = "C:\path\to\output\log"
$env:TRAIN_TF_EVENTS_PATH = "C:\path\to\output\tf_events"

python -u .\train.py `
  --ns_tokenizer_type rankmixer `
  --user_ns_tokens 5 `
  --item_ns_tokens 2 `
  --num_queries 2 `
  --ns_groups_json "" `
  --emb_skip_threshold 1000000 `
  --num_workers 0 `
  --num_epochs 3 `
  --batch_size 128
```

如果只是想快速检查代码能不能跑通，可以先用更小的配置：

```bash
bash run.sh \
  --num_epochs 1 \
  --batch_size 16 \
  --num_workers 0 \
  --train_ratio 0.05 \
  --valid_ratio 0.2 \
  --seq_max_lens seq_a:64,seq_b:64,seq_c:128,seq_d:128
```

## 核心参数

| 参数 | 作用 | 常用建议 |
| --- | --- | --- |
| `--data_dir` / `TRAIN_DATA_PATH` | 训练数据目录 | 放官方 `*.parquet` 和 `schema.json` |
| `--batch_size` | batch 大小 | 显存不够就先降到 16/32/64 |
| `--num_workers` | DataLoader worker 数 | Windows 或调试时用 0；Linux 服务器可用 4/8/16 |
| `--valid_ratio` | 用尾部 Row Groups 做验证集的比例 | 默认 0.1 |
| `--train_ratio` | 只使用前 N% 训练 Row Groups | 快速 smoke test 可用 0.01 到 0.1 |
| `--seq_max_lens` | 各序列域最大长度 | 长序列很耗显存，优先调这里 |
| `--d_model` | 主干 hidden size | 默认 64 |
| `--num_queries` | 每个序列域生成几个 query token | `run.sh` 默认 2 |
| `--seq_encoder_type` | 序列编码器 | `transformer` 精度优先，`swiglu` 更快，`longer` 做 Top-K 压缩 |
| `--rank_mixer_mode` | token mixing 模式 | 默认 `full`，若维度约束报错可临时用 `ffn_only` |
| `--loss_type` | 损失函数 | `bce` 或 `focal` |
| `--emb_skip_threshold` | 超高基数特征不建 embedding | 显存紧张时有用 |

`rank_mixer_mode=full` 有一个硬约束：

```text
d_model % T == 0
T = num_queries * num_sequences + num_ns
```

如果启动时报 `d_model must be divisible by T`，可以从三处修：

1. 改 `--d_model`，让它能整除 `T`
2. 改 `--num_queries`、`--user_ns_tokens`、`--item_ns_tokens`
3. 临时用 `--rank_mixer_mode ffn_only`

## 代码主线解析

### `train.py`

`train.py` 是唯一训练入口：

1. 解析命令行参数和环境变量
2. 找到 `data_dir/schema.json`
3. 调用 `get_pcvr_data()` 创建 train/valid DataLoader
4. 根据 `ns_groups.json` 或默认单特征分组创建 NS groups
5. 用数据集 schema 推导特征规格，初始化 `PCVRHyFormer`
6. 创建 `PCVRHyFormerRankingTrainer` 开始训练

环境变量优先级高于命令行参数：

```text
TRAIN_DATA_PATH
TRAIN_CKPT_PATH
TRAIN_LOG_PATH
TRAIN_TF_EVENTS_PATH
```

### `dataset.py`

`PCVRParquetDataset` 是一个 `IterableDataset`，核心工作是把 Parquet 的多列结构变成模型 batch：

- 按 Row Group 切分训练集/验证集，验证集取尾部 Row Groups
- `user_int` / `item_int`：标量或多值离散特征补齐成 `[B, dim]`
- `user_dense`：float array 补齐后拼成 `[B, total_dense_dim]`
- `seq`：每个序列域单独处理，输出 `[B, sideinfo_count, max_len]`
- timestamp 序列会被转成时间差 bucket，再通过模型里的 `time_embedding` 使用
- 小于等于 0 的 id 当 padding；超过 vocab 的 id 默认裁剪成 0，并记录 OOB 统计

### `model.py`

模型主体是 `PCVRHyFormer`：

- `GroupNSTokenizer`：按 `ns_groups.json` 把离散特征组投影成 NS tokens
- `RankMixerNSTokenizer`：先拼接所有离散特征 embedding，再切成固定数量的 NS tokens，`run.sh` 默认使用它
- 序列侧每个 domain 会独立 embedding，然后通过 `transformer/swiglu/longer` 编码
- `MultiSeqQueryGenerator` 为每个序列域生成 query tokens
- `MultiSeqHyFormerBlock` 做 sequence evolution、query decoding 和 RankMixer token fusion
- 最后拼接所有 query tokens，通过 MLP 输出一个 PCVR logit

论文角度看，`PCVRHyFormer.forward()` 可以分成 5 步：

1. 离散和 dense 非序列特征被投影成 NS/global tokens
2. 每个序列域的 side-info 特征被 embedding 后投影成序列 token
3. `MultiSeqQueryGenerator` 用 NS tokens 和各序列 mean pooling 生成 per-sequence queries
4. 多层 `MultiSeqHyFormerBlock` 反复执行 Query Decoding + Query Boosting
5. 顶层 query tokens 拼接后进入 MLP classifier，输出二分类 logit

这也是论文强调的“bidirectional, layer-wise interaction”：不是先把序列压成一个固定向量再做交互，而是每层都让全局 query 去读序列，再把读出来的 query 和非序列 token 做混合。

### `trainer.py`

Trainer 虽然名字里有 `Ranking`，但实际是 pointwise 二分类：

- loss：`BCEWithLogitsLoss` 或 `sigmoid_focal_loss`
- metric：Binary AUC 和 binary logloss
- 优化器：Embedding 参数用 `Adagrad`，非 Embedding 参数用 `AdamW`
- 梯度裁剪：`max_norm=1.0`
- checkpoint：只保留当前 best model 目录，目录里会带 `model.pt`、`schema.json`、`train_config.json`，如果使用了分组配置也会复制 `ns_groups.json`

## 输出文件

训练后一般会看到：

```text
ckpt/
└── global_step123.layer=2.head=4.hidden=64.best_model/
    ├── model.pt
    ├── schema.json
    ├── train_config.json
    └── ns_groups.json  # 使用 group 配置时才有

log/
└── train.log

tf_events/
└── events.out.tfevents.*
```

查看 TensorBoard：

```bash
tensorboard --logdir /path/to/output/tf_events
```

## 常见问题

### `Path(args.ckpt_dir)` 或 `Path(args.tf_events_dir)` 报错

说明输出目录没有传进来。设置：

```bash
export TRAIN_CKPT_PATH=/path/to/ckpt
export TRAIN_LOG_PATH=/path/to/log
export TRAIN_TF_EVENTS_PATH=/path/to/tf_events
```

### Windows 上 `run.sh` 跑不了

`run.sh` 是 bash 脚本。Windows 可以用 Git Bash/WSL，或者直接参考上面的 PowerShell 命令跑 `python -u .\train.py`。

### 显存不够

优先尝试：

```text
--batch_size 调小
--seq_max_lens 调小
--seq_encoder_type swiglu
--emb_skip_threshold 设置为较低阈值
```

### AUC 一直是 0

`trainer.py` 在验证集只有单一 label 时会把 AUC 置为 0。先检查 `valid_ratio` 切出来的 Row Groups 是否同时包含正负样本。

## 论文启发的实验方向

论文实验里有几个结论可以直接转成比赛 baseline 的调参方向：

- Query 信息很重要：论文消融显示去掉 sequence pooling tokens 或 non-seq/global 信息都会掉 AUC。对应到代码里，可以重点试 `--num_queries`、`--user_ns_tokens`、`--item_ns_tokens`，以及 `ns_groups.json` 的语义分组。
- 多序列不要随便 merge：论文发现把多序列强行合并会损失不同序列的语义差异。baseline 已经按 `seq_a/seq_b/seq_c/seq_d` 独立处理，改代码时尽量保留这个结构。
- 交互层和序列层要一起看：论文的增益来自 Query Decoding 和 Query Boosting 的交替堆叠。对应参数是 `--num_hyformer_blocks`、`--rank_mixer_mode`、`--seq_encoder_type`。
- 序列 side information 越丰富，HyFormer 越能吃到收益：论文的 scaling 分析显示，扩展序列 sparse dim 对 HyFormer 的收益比传统 LONGER + RankMixer 更明显。比赛里可以优先研究 `schema.json` 里序列特征的保留、截断和高基数 embedding 跳过策略。
- 计算预算要守住：论文强调在参数/FLOPs 可比条件下胜过 LONGER + RankMixer。baseline 里最容易爆显存的是 embedding 表和序列长度，优先调 `--emb_skip_threshold`、`--seq_max_lens`、`--seq_encoder_type`。

## 建议的改进入口

- 想改特征：先看官方 `schema.json`，再改 `ns_groups.json` 或 `--seq_max_lens`
- 想省显存：优先动 `batch_size`、序列长度、`emb_skip_threshold`
- 想改模型：从 `model.py` 的 `PCVRHyFormer.forward()` 和 `MultiSeqHyFormerBlock` 入手
- 想改 loss 或评估：改 `trainer.py` 的 `_train_step()`、`evaluate()`
- 想做快速实验：固定 `--train_ratio 0.05 --num_epochs 1 --num_workers 0`，确认能跑通再放大
