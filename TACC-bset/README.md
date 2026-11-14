# TAAC 代码改动说明

当前版本主要做了的处理：

1. 使用 AMP + compile 加快训练
2. 增加严格时间切分开关，避免时间穿越
3. 样本级时间日历特征（UTC+8）
4. User-Item Aware Query Generator 与 User-Item Shortcut Residual 消融
5. Field-Aware Dense Tokenizer（field_attn）
6. Sparse Adagrad cold restart 状态恢复修复
7. FirstDecodedQResidualBranch 消融
8. Target Attention / DIN
9. 扩大序列截断上限
10. 序列时间统计特征（seq_time_stats）

当前 `run.sh` 默认新增并显式带上的参数，以及对应作用如下（按修改编号排序）：

| 参数 | 对应修改 | 作用 |
| --- | --- | --- |
| `--ns_tokenizer_type rankmixer` | `0` | 默认使用 `RankMixer` 版 NS tokenizer，而不是依赖 `ns_groups.json` 的分组方案。 |
| `--user_ns_tokens 5` | `0` | 为用户侧 NS 特征分配 5 个 token。 |
| `--item_ns_tokens 2` | `0` | 为物料侧 NS 特征分配 2 个 token。 |
| `--num_queries 2` | `0` | 默认使用 2 个 query token，与当前 `RankMixer` 配置匹配。 |
| `--ns_groups_json ""` | `0` | 显式置空 `ns_groups.json` 路径，表示当前默认配置不走 group tokenizer。 |
| `--emb_skip_threshold 1000000` | `0` | 不跳过任何有 vocab 的特征 embedding，让所有普通特征都真正参与训练与 cold restart。 |
| `--num_workers 8` | `0` | 将 DataLoader worker 数设置为 8，平衡数据加载吞吐和机器资源占用。 |
| `--use_amp` | `1` | 启用 AMP 自动混合精度训练，减少训练开销并提升吞吐。 |
| `--amp_dtype auto` | `1` | 自动选择 AMP 精度类型，优先使用 `bf16`，不支持时回退到 `fp16`。 |
| `--use_compile` | `1` | 启用 `torch.compile()`，进一步优化训练图执行效率。 |
| `--compile_mode default` | `1` | `torch.compile()` 支持 `default`、`reduce-overhead`、`max-autotune` 三种模式，当前默认使用 `default` 并关闭 CUDA Graph。 |
| `--compile_disable_cudagraphs` | `1` | 关闭 Inductor CUDA Graph capture，避免 `reduce-overhead` 的 CUDA Graph Trees 私有内存池问题。 |
| `--use_calendar_time_features` | `3` | 默认开启样本级时间日历特征，使用根级 `timestamp` 生成 `weekday/hour` 特征。 |
| `--query_generator_type ui_aware` | `4` | 默认启用 item-aware query generator，让 query 更直接围绕"当前 item 应该去历史里检索什么"生成。 |
| `--ui_shortcut_type bi_cross_attn` | `4` | 默认启用双向 user-item cross-attention shortcut，在分类头前补充浅层 user-item 匹配信号。 |
| `--ui_shortcut_alpha_init 1.0` | `4` | 设置 `ui_shortcut` 分支的可学习 residual scale 初始值为 `1.0`，用于控制该旁路初始时的注入强度。 |
| `--dense_tokenizer_type field_attn` | `5` | 默认启用 field-aware dense tokenizer，按特征字段独立投影后用 attention 聚合。 |
| `--reinit_sparse_after_epoch 1` | `6` | 从第 1 个 epoch 结束后开始，每轮执行一次 sparse embedding cold restart。 |
| `--reinit_cardinality_threshold 0` | `6` | 对所有 `vocab_size > 0` 的普通特征 embedding 执行重置。 |
| `--first_decoded_shortcut_type mixer` | `7` | 默认启用第一层 decoded-q residual 分支，并使用 `RankMixerBlock(mode="full")` 做分支内轻量混合。 |
| `--first_decoded_alpha_init 0.2` | `7` | 将当前 `run.sh` 默认实验的 decoded-q residual 初始缩放系数设为 `0.2`。 |
| `--target_attn_type din` | `8` | 默认启用 DIN-style target attention residual，让当前 item 直接对各路历史序列做目标相关聚合。 |
| `--seq_max_lens "seq_a:512,seq_b:512,seq_c:512,seq_d:1024"` | `9` | 扩大各序列 domain 的截断上限，让模型看到更长历史序列（seq_d 是 seq_a/b/c 的两倍）。 |
| `--use_seq_time_stats` | `10` | 默认开启序列时间统计特征，从 `seq_a/seq_b/seq_c/seq_d` 各序列的 `timestamp` 列派生出最近 5 档（30min/1h/6h/24h/7d）内的行为数量，作为序列级时间上下文向量。 |

当前默认不启用、但已经支持的功能参数如下（按修改编号排序）：

| 参数 | 对应修改 | 作用 |
| --- | --- | --- |
| `--strict_time_split` | `2` | 启用严格时间切分，按全局 `timestamp` 排序后再划分 train/valid，避免时间穿越；默认不启用，保持原有的 Row Group 切分方式。 |
| `--query_generator_type baseline` | `4` | 切回原始 `MultiSeqQueryGenerator`，用于和 `ui_aware` 做干净对照。 |
| `--ui_shortcut_type none` | `4` | 关闭 user-item shortcut residual，只保留主干 HyFormer 预测路径。 |
| `--dense_tokenizer_type single` | `5` | 切回原始 dense tokenizer，把全部 dense 特征整体投影为 1 个 token。 |
| `--emb_skip_threshold 1000000` | `6` | 跳过超高基数 embedding，减少显存占用；适合显存紧张时回退使用。 |
| `--first_decoded_shortcut_type none` | `7` | 关闭第一层 decoded-q residual 分支，恢复没有该旁路的 baseline 行为。 |
| `--first_decoded_shortcut_type ffn_only/mlp` | `7` | 切换 decoded-q residual 分支的消融形式，只保留 FFN 或纯 MLP 路径。 |
| `--first_decoded_detach` | `7` | 对第一层 `decoded_qs` 做 detach，让该旁路不把额外梯度回传到第一层 HyFormerBlock。 |
| `--target_attn_type none` | `8` | 关闭 Target Attention / DIN 分支，只保留主干与已有 shortcut 路径。 |
| `--target_attn_alpha_init 0.1` | `8` | 设置 Target Attention / DIN 分支的可学习 residual scale 初始值。 |
| `--seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"` | `9` | 回退到更保守的序列截断上限，减少显存占用和训练时间。 |
| `--no_seq_time_stats` | `10` | 关闭序列时间统计特征，节省序列侧计算开销。 |
## 1. AMP + compile

### 解决了什么问题

原始训练流程在 GPU 上默认使用常规精度训练，训练速度和显存占用都不够理想。

这次处理的目标是：

- 加快训练速度
- 尽量利用 CUDA 的混合精度能力
- 在不改动模型主逻辑的前提下，降低部分显存压力

### 修改了什么

这次在训练入口里增加了 AMP / compile 相关参数：

- `--use_amp`
- `--amp_dtype`
- `--use_compile`
- `--compile_mode`
- `--compile_disable_cudagraphs`

其中：

- `--use_amp` 用来控制是否启用自动混合精度
- `--amp_dtype` 用来控制混合精度的数据类型，支持 `auto`、`fp16`、`bf16`
- `--use_compile` 用来控制是否启用 `torch.compile()`
- `--compile_mode` 用来控制 `torch.compile()` 的模式，支持 `default`、`reduce-overhead`、`max-autotune`
- `--compile_disable_cudagraphs` 用来禁用 Inductor CUDA Graph capture

同时，`run.sh` 已经默认带上了：

```bash
--use_amp \
--amp_dtype auto \
--use_compile \
--compile_mode default \
--compile_disable_cudagraphs
```

也就是说，当前通过 `run.sh` 启动时，默认就是 `AMP + compile` 一起开启的。

AMP 的处理方式是：

- 在训练时开启 CUDA autocast
- `amp_dtype=auto` 时优先选择 `bf16`
- 如果当前环境不支持 `bf16`，则回退到 `fp16`

compile 的处理方式是：

- 在模型 `.to(device)` 之后调用 `torch.compile()`
- 训练入口参数默认值是 `default`
- 当前 `run.sh` 默认显式使用 `default`，并关闭 Inductor CUDA Graph capture
- 也可以通过 `--compile_mode` 切到 `reduce-overhead` 或 `max-autotune`

这里的设计重点是让 compile 和 AMP 共同作用，而不是二选一：

- AMP 负责降低训练时的计算精度成本
- compile 负责让前向 / 反向图有机会被进一步优化
- 两者一起开时，通常更容易拿到更好的吞吐

这样做的好处是：

- 新一些的 GPU 可以优先利用 `bf16` 的稳定性和速度
- 老一些或不支持 `bf16` 的环境仍然可以退回 `fp16`
- 在支持 `torch.compile()` 的 PyTorch 环境里，可以进一步优化训练吞吐
- 不需要为不同机器手动改训练脚本

### `compile_mode` 怎么选

这三个模式可以简单理解为：

- `default`：通用默认值，适合先验证训练是否稳定
- `reduce-overhead`：更保守，适合短实验或频繁调参
- `max-autotune`：更激进，适合长时间正式训练

当前仓库在启用 `field_attn` dense tokenizer 时，把 `run.sh` 默认设成 `default + --compile_disable_cudagraphs`，主要是为了避免 `reduce-overhead` 的 CUDA Graph Trees 私有内存池问题，以及 `max-autotune` 的 Triton benchmark 峰值内存。

因此可以按下面的方式理解：

- `field_attn` 正式长跑训练：优先用 `default + --compile_disable_cudagraphs`
- 日常小实验、只跑少量 step / 少量 epoch：优先用 `default`
- 如果只是想尽量减少 compile 本身的额外开销，再考虑 `reduce-overhead`


## 2. 严格时间切分开关

### 解决了什么问题

原来的训练集和验证集划分方式不是严格按样本时间先后切分的，因此会存在时间穿越问题：

- 训练数据里可能看到比验证样本更晚的行为
- 这样会让验证结果偏乐观

这次处理的目标是：

- 让 train / valid 按样本级 `timestamp` 严格切分
- 尽量避免验证集泄漏未来信息

### 是如何处理的

严格时间切分的实现逻辑是：

1. 读取所有样本的 `timestamp`
2. 按样本级时间全局排序
3. 再划分 train / valid
4. 如果切分边界正好落在同一个 `timestamp` 的样本块中间，则把这个边界时间戳块整体划到验证集

这样做是为了保证：

```text
train_timestamp_max < valid_timestamp_min
```

也就是说，训练集中的最晚时间一定早于验证集中的最早时间。

另外还有两个行为需要说明：

- 缺失 `timestamp` 的样本不会参与严格时间切分
- 如果边界时间戳块太大，移动后会导致训练集为空，则代码会直接报错，而不会继续做伪严格切分



### 常用运行方式

默认运行，不包含时间切分：

```bash
bash run.sh
```

默认 AMP + 严格时间切分：

```bash
bash run.sh --strict_time_split
```
## 3. 样本级时间日历特征（UTC+8）

### 解决了什么问题？

原先代码里，`timestamp` 主要用于两件事：

- 严格时间切分 train / valid
- 计算序列行为相对当前样本时间的 `time_bucket`

但这还没有显式利用"当前样本发生在一周中的哪一天、一天中的哪个小时"这类日历信息。
这次新增了一条独立的样本级时间特征链路：

- `timestamp_weekday`
- `timestamp_hour`

用于把样本发生时的本地时间模式直接送入模型。

### 现在是如何处理的？

对于每条样本的根 `timestamp`，现在会额外派生两个离散特征：

- `timestamp_weekday`
  - 含义：星期几，不是"第几周"
  - 编码：`Monday=1, ..., Sunday=7`
  - `0` 预留给缺失或非法时间戳
- `timestamp_hour`
  - 含义：小时
  - 编码：`00:00=1, ..., 23:00=24`
  - `0` 预留给缺失或非法时间戳

这两个特征现在按 `UTC+8` 计算。

例如：

```text
1770565361 -> 2026-02-08 23:42:41 (UTC+8)
weekday_id = 7
hour_id = 24
```

### week / hour 的判断逻辑

这里没有在数据管道里逐条调用 `datetime` 一类日期库，而是使用纯整数运算完成，保证训练和推理口径完全一致，同时避免额外开销。

计算逻辑为：

```text
local_ts = timestamp + 8 * 3600
hour = (local_ts // 3600) % 24
weekday = ((local_ts // 86400) + 3) % 7
```

然后再映射成送入模型的 id：

```text
hour_id = hour + 1
weekday_id = weekday + 1
```

其中 `+3` 的原因是：

- Unix epoch 的 `1970-01-01` 是星期四
- 按常用 weekday 约定，`Monday=0, ..., Sunday=6`
- 星期四对应 `3`

所以可以直接通过固定偏移得到 weekday，而不需要额外依赖日历库。

### 模型里如何使用？

这次新增的 `weekday/hour` 特征与原有 `time_bucket` 是并存关系：

- `time_bucket`：描述"历史序列行为距离当前样本过去了多久"
- `weekday/hour`：描述"当前样本本身发生在本地什么时间"

模型侧会为这两个新特征分别建立 embedding：

- `weekday_embedding`
- `hour_embedding`

然后把它们合成为一个样本级时间上下文向量，并加到现有 `ns_tokens` 上。

这样做的好处是：

- 不增加 token 数
- 不破坏现有 HyFormer 结构
- 不影响 `d_model % T == 0` 的约束

### 训练侧与推理侧是否一致？

是一致的，这次改动已经同时同步到：

- 训练侧 `dataset.py / model.py / trainer.py / train.py`
- 推理侧 `evaluation/dataset.py / evaluation/model.py / evaluation/infer.py`

也就是说：

- 训练时会生成并使用 `timestamp_weekday` / `timestamp_hour`
- 推理时也会生成并使用完全相同的特征
- checkpoint 中的 `train_config.json` 也会记录相关配置，便于 `evaluation/infer.py` 正确恢复模型结构

### 新增开关

训练入口新增了两个参数：

- `--use_calendar_time_features`
- `--no_calendar_time_features`

默认开启

因此当前默认训练包含：

- 原有的序列 `time_bucket`
- 新增的 `UTC+8 weekday/hour` 样本级时间特征


## 4. User-Item Aware Query Generator 消融选项

### 解决了什么问题？

原模型的 query 生成逻辑位于 `model.py` 的 `MultiSeqQueryGenerator` 中，核心形式是：

```text
ns_tokens + MeanPool(seq_i) -> query_i
```

也就是说，query 在进入每个序列 domain 的 cross-attention 之前，会同时看到：

- 用户 / 商品非序列 token 拼成的 `ns_tokens`
- 当前 domain 序列的 masked mean pooling

这套逻辑可以工作，但对于 CVR 任务来说，query 的目标语义还不够直接。CVR 强依赖当前商品，历史行为中真正重要的是：

```text
哪些历史行为和当前 item 有关？
```

因此新增了一个完整替换式的 `UserItemAwareQueryGenerator`，用于消融实验。它让当前 item 先主动查询用户非序列特征，得到"当前商品视角下的用户表示"，再结合每个序列 domain 自己的 masked pooling 生成 query。

### 新增了什么？

训练入口新增参数：

```bash
--query_generator_type baseline|ui_aware
```

含义如下：

- `baseline`：默认值，保持原始 `MultiSeqQueryGenerator`
- `ui_aware`：启用新的 `UserItemAwareQueryGenerator`

默认行为不变：

```bash
bash run.sh
```

等价于：

```bash
bash run.sh --query_generator_type baseline
```

启用新的 user-item aware query generator：

```bash
bash run.sh --query_generator_type ui_aware
```

### 新 query generator 的数据流

新增逻辑不会把 old query 和 new query 相加，也不是 residual 增强，而是完整替换 query 生成路径。

整体流程为：

```text
user_int_feats / user_dense_feats
        ↓
    user_tokens

item_int_feats / item_dense_feats
        ↓
    item_tokens

item_tokens 作为 Q
user_tokens 作为 K/V
        ↓
item -> user cross-attention
        ↓
item_aware user context
        ↓
item_repr + user_given_item
        ↓
ui_base
        ↓
ui_base + MeanPool(seq_i) + interaction
        ↓
domain-specific query_i
        ↓
query_i cross-attention seq_i
```

其中 `interaction` 包含：

```text
ui_base * seq_pooled_i
ui_base - seq_pooled_i
```

这样做的目标是保留每个序列 domain 的样本级信息，同时让 query 的主语义变成：

```text
当前 item 视角下，应该去用户历史序列里检索什么证据？
```

### 模型侧改动

训练侧改动包括：

- `model.py`
  - 新增 `UserItemAwareQueryGenerator`
  - `PCVRHyFormer` 新增 `query_generator_type` 参数
  - 新增 `_build_nonseq_tokens()`，显式拆分：
    - `user_tokens`
    - `item_tokens`
    - `ns_tokens`
  - `forward()` / `predict()` 根据 `query_generator_type` 选择 query 生成器
- `train.py`
  - 新增 `--query_generator_type`
  - 将该参数写入 `model_args`
  - checkpoint 的 `train_config.json` 会记录该配置

推理侧同步改动包括：

- `evaluation/model.py`
  - 与训练侧 `model.py` 保持一致
- `evaluation/infer.py`
  - `_FALLBACK_MODEL_CFG` 新增：

```python
'query_generator_type': 'baseline'
```

这样评测时如果 checkpoint 中存在 `train_config.json`，会按训练时保存的 `query_generator_type` 恢复模型结构。

### 和原始 query 的区别

原始 query：

```text
query_i = FFN(ns_tokens_flat, seq_mean_i)
```

新增 `ui_aware` query：

```text
user_given_item = CrossAttention(Q=item_tokens, K=user_tokens, V=user_tokens)
ui_base = MLP(item_repr, user_given_item, item_repr * user_given_item, item_repr - user_given_item)
query_i = MLP(ui_base, seq_mean_i, ui_base * seq_mean_i, ui_base - seq_mean_i)
query_i = query_i + query_id_emb + domain_emb
```

其中：

- `query_id_emb` 用于区分同一个 domain 下的多个 query
- `domain_emb` 用于保留 `seq_a / seq_b / seq_c / seq_d` 的不同检索方向
- `seq_mean_i` 继续保留每个 domain 的样本级序列摘要

### 推荐实验方式

建议先跑一组最干净的对照：

```bash
bash run.sh --query_generator_type baseline
bash run.sh --query_generator_type ui_aware
```

如果要和其他已有消融组合，也可以直接追加参数：

```bash
bash run.sh \
  --query_generator_type ui_aware \
  --no_calendar_time_features
```

或者：

```bash
bash run.sh \
  --query_generator_type ui_aware \
  --rank_mixer_mode ffn_only
```

### 注意事项

- `ui_aware` 是结构性改动，会改变 checkpoint 参数名和模型结构。
- 使用 `ui_aware` 训练出的 checkpoint，评测时必须使用同步后的 `evaluation/model.py` 和 `evaluation/infer.py`。
- 如果缺少 `train_config.json`，推理侧 fallback 会按 `baseline` 构建模型；因此正式评测时应确保 checkpoint 目录中包含训练保存的 `train_config.json`。

### User-Item Shortcut Residual 消融

在 query generator 之外，新增一个独立的预测头前 user-item 交互旁路，训练入口参数为：

```bash
--ui_shortcut_type none|bi_cross_attn
--ui_shortcut_alpha_init 1.0
```

含义如下：

- `none`：默认值，不启用旁路，保持原预测路径
- `bi_cross_attn`：启用双向 user-item cross-attention shortcut
- `ui_shortcut_alpha_init`：UI shortcut residual 的可学习缩放系数初始值，默认 `1.0`

该旁路不使用原始整数 ID，而是复用 tokenizer / dense projection 后的：

```text
user_tokens
item_tokens
```

这里和 `ui_aware` query generator 的关系需要区分清楚：

```text
输入 token：共用 user_tokens / item_tokens
注意力参数：不共用
query generator：单向 item_tokens -> user_tokens
shortcut residual：双向 user_tokens <-> item_tokens
作用位置：query generator 在序列 cross-attention 前生成 query；
          shortcut residual 在 HyFormer final Q representation 后、classifier 前融合
```

也就是说，二者共享的是同一批 tokenizer 后的用户 / 商品 token 表示，但不是共享同一个 attention 模块。

`ui_aware` query generator 只做：

```text
item_tokens as Q
user_tokens as K/V
```

它的输出主要服务于"当前 item 视角下，应该生成什么 query 去检索用户历史序列"。

`bi_cross_attn` shortcut 旁路则做双向 cross-attention：

```text
user_tokens as Q, item_tokens as K/V
item_tokens as Q, user_tokens as K/V
```

它的输出不参与 query 生成，而是作为预测头前的 user-item matching residual。

结构为：

```text
user_tokens as Q, item_tokens as K/V -> user-aware item signal
item_tokens as Q, user_tokens as K/V -> item-aware user signal
两者 mean pooling + interaction MLP -> ui_residual
gate = sigmoid(MLP(final_q_output, ui_residual))
final_output = final_q_output + ui_shortcut_alpha * gate * ui_residual
```

其中 `final_q_output` 是 HyFormer 主干经过 `output_proj` 后的最终 Q representation。这样主干仍然负责序列建模和 Q token 表征，旁路只补充用户 / 商品非序列浅层匹配信号。

这里的 `ui_shortcut_alpha` 是一个可学习的标量参数，初始值由 `--ui_shortcut_alpha_init` 控制。它的作用是在 `gate` 之外再对整个 UI shortcut residual 做一层全局缩放，便于控制该旁路刚开始训练时对主干表示的扰动强度。

- 如果希望该旁路一开始就保持较强作用，可以使用 `--ui_shortcut_alpha_init 1.0`
- 如果希望更平滑地引入该旁路，可以尝试更小的初始值，例如 `0.1` 或 `0.01`

启用方式：

```bash
bash run.sh \
  --ui_shortcut_type bi_cross_attn \
  --ui_shortcut_alpha_init 1.0
```

也可以和 `ui_aware` query generator 一起使用：

```bash
bash run.sh \
  --query_generator_type ui_aware \
  --ui_shortcut_type bi_cross_attn \
  --ui_shortcut_alpha_init 1.0
```

推理侧已同步：

```python
'ui_shortcut_type': 'none'
'ui_shortcut_alpha_init': 1.0
```

如果 checkpoint 中有 `train_config.json`，评测会按训练时保存的 `ui_shortcut_type` 和 `ui_shortcut_alpha_init` 恢复模型结构。对于旧 checkpoint，如果缺少 `ui_shortcut_alpha` 权重，推理侧会回退到模型初始化时的默认值。


## 5. Field-Aware Single Dense Tokenizer

### 解决了什么问题？

原始 dense 特征处理方式是将全部 dense 特征整体 `Linear -> LayerNorm -> SiLU` 后压成 1 个 dense token。这种方式不区分不同 dense 字段的语义差异，所有字段被平等对待。

### 修改了什么？

新增 `--dense_tokenizer_type single|field_attn`：

- `single`：保留原始路径，把全部 dense 特征整体 `Linear -> LayerNorm -> SiLU` 后压成 1 个 dense token。
- `field_attn`：按 `FeatureSchema.entries` 中的 `(feature_id, offset, length)` 切分 dense 特征；每个 fid 独立 `Linear(length, d_model) -> LayerNorm -> SiLU -> Dropout`，再加 field id embedding，最后用 1 个 learnable query 通过 `RoPEMultiheadAttention` 聚合成 1 个 dense token。

`field_attn` 的输出仍然是 `[B, 1, d_model]`，所以接入位置不变：

```text
user_parts = [user_ns, user_dense_token]
item_parts = [item_ns, item_dense_token]
ns_tokens = cat(user_tokens, item_tokens)
```

在当前默认配置下，token 数仍保持不变：`user_ns=5`，`user_dense_token=1`，`item_ns=2`，`num_queries=2`，`num_sequences=4`，因此 `T = 2*4 + 8 = 16`，不会改变 RankMixer 在 `rank_mixer_mode=full` 下对 `d_model % T == 0` 的约束。

### compile 配合说明

`field_attn` 默认配合：

```bash
--compile_mode default \
--compile_disable_cudagraphs
```

原因是 `field_attn` 会为每个 dense fid 建一个独立投影，小矩阵乘数量明显多于原始 `single` 路径；如果继续使用 `max-autotune`，Triton 会对这些 matmul 做大量 autotune benchmark，容易在受限容器里被系统 kill，典型表现是 `exit code 137`。另外，`reduce-overhead` 会启用 CUDA Graph Trees，可能触发 `_cuda_setCheckpointPoolState` allocator 错误。

### 模型侧改动

- `model.py` / `evaluation/model.py` 新增 `FieldAwareSingleDenseTokenizer`，并在 `PCVRHyFormer` 中按 `dense_tokenizer_type` 切换 dense 路径。
- `train.py` 会把 `user_dense_schema.entries`、`item_dense_schema.entries` 和 `dense_tokenizer_type` 传入模型，并写入 checkpoint 的 `train_config.json`。
- `evaluation/infer.py` 会从 `train_config.json` 恢复 `dense_tokenizer_type`，并用评测数据集的 dense schema entries 构造同样的模型。

### 推荐实验命令

```bash
# 使用原始 single dense tokenizer
bash run.sh --dense_tokenizer_type single

# 使用 field_attn dense tokenizer（当前默认）
bash run.sh --dense_tokenizer_type field_attn
```


## 6. Sparse Adagrad Cold Restart 状态恢复修复

修改者：`yuyu`

### 问题背景

`trainer.py` 在每个 epoch 结束后会对 sparse feature Embedding 执行 cold restart：

1. 保存旧 `sparse_optimizer` 的 Adagrad state
2. 调用 `model.reinit_high_cardinality_params(...)` 重置满足阈值条件的 Embedding
3. 重新创建 sparse Adagrad optimizer
4. 恢复未被重置的低基数 Embedding 的 optimizer state

原逻辑在保存旧 state 时使用了：

```python
p.data_ptr() in self.sparse_optimizer.state
```

但 PyTorch optimizer 的 `state` 通常以 `Parameter` 对象本身作为 key，而不是以 `data_ptr()` 整数作为 key。因此旧逻辑可能导致 `old_state` 为空，低基数 Embedding 的 Adagrad 累积状态没有真正恢复。

### 修复内容

当前逻辑已经改为：

- 使用 `Parameter` 对象 `p` 作为 `old_state` 的 key
- 只迁移未被重置参数的旧 state 引用，不再 `copy.deepcopy` 全量 Adagrad state
- 继续使用 `reinit_ptrs: set[int]` 判断哪些 Embedding 权重被重新初始化
- 重建 sparse Adagrad optimizer 后：
  - `p.data_ptr()` 在 `reinit_ptrs` 中：说明是本轮重置参数，不恢复旧 state
  - `p.data_ptr()` 不在 `reinit_ptrs` 中：说明是低基数保留参数，尝试恢复旧 state

核心逻辑为：

```python
old_state = {}
reinit_ptrs = self.model.reinit_high_cardinality_params(
    self.reinit_cardinality_threshold
)

for group in self.sparse_optimizer.param_groups:
    for p in group['params']:
        if p.data_ptr() in reinit_ptrs:
            continue
        if p in self.sparse_optimizer.state:
            old_state[p] = self.sparse_optimizer.state[p]

sparse_params = self.model.get_sparse_params()
self.sparse_optimizer = torch.optim.Adagrad(
    sparse_params,
    lr=self.sparse_lr,
    weight_decay=self.sparse_weight_decay,
    foreach=False,
)

restored = 0
reset = 0
for p in sparse_params:
    if p.data_ptr() in reinit_ptrs:
        reset += 1
        continue
    if p in old_state:
        self.sparse_optimizer.state[p] = old_state[p]
        restored += 1
```

### 新增日志

重建 sparse Adagrad optimizer 后会输出：

```text
reset_embedding=...
restored_embedding=...
```

用于确认：

- 本轮被重置的 Embedding 是否被识别为 reset
- 未重置 Embedding 的 Adagrad state 是否恢复成功

### 注意事项

- 没有修改 `dense_optimizer` 逻辑
- 没有修改 `model.reinit_high_cardinality_params()` 的返回类型，仍然返回 `set[int]`
- 该修复只影响 sparse Embedding cold restart 后的 Adagrad state 恢复行为

### 当前默认 cold restart 配置

`run.sh` 当前显式使用全量 sparse feature cold restart：

```bash
--reinit_sparse_after_epoch 1 \
--reinit_cardinality_threshold 0
```

含义是：从第 1 个 epoch 结束后开始，每个 epoch 都对所有 `vocab_size > 0` 的用户 / 商品 / 序列特征 Embedding 执行重置，并重建 sparse Adagrad optimizer。

注意：`time_embedding`、`weekday_embedding`、`hour_embedding` 属于时间辅助 Embedding，`model.reinit_high_cardinality_params(...)` 中会保留它们，不参与这个特征 Embedding 冷启动。

由于 `reinit_cardinality_threshold=0` 会让所有已创建的普通特征 Embedding 都被重置，理论上低基数特征 Embedding 不再有需要恢复的 Adagrad state；日志里的 `restored_embedding` 通常只会对应未重置的时间辅助 Embedding。

为了让"所有 `vocab_size > 0` 的特征"都真正创建 Embedding 并参与每轮冷启动，当前 `run.sh` 还显式设置：

```bash
--emb_skip_threshold 1000000
```

这表示不跳过任何有 vocab 信息的特征。需要注意，这会比 `--emb_skip_threshold 1000000` 占用更多显存；如果遇到显存不足，可以临时退回：

```bash
bash run.sh --emb_skip_threshold 1000000
```


## 7. FirstDecodedQResidualBranch 消融

### 目标

新增一条 classifier 前的早期检索证据 residual 分支：

```text
第一层 MultiSeqHyFormerBlock 内部 decoded_qs
    -> FirstDecodedQResidualBranch
    -> gated residual
    -> classifier 前 final representation
```

这里的 `decoded_qs` 是第一层 block 中 `Q token` 对各自序列做 cross-attention 后、还没有和 NS tokens 进入 `RankMixerBlock` 混合前的直接结果。它不是初始 `q_tokens_list`，也不是 RankMixer 输出后的 `next_q_list`。

注意：这个分支不是把 raw `decoded_qs` 原样直接加到最终表示里。当前默认 `mixer` 模式会先在 `FirstDecodedQResidualBranch` 内部对 `decoded_qs` 单独做一次轻量演变：

```text
decoded_qs
    -> concat over sequence domains
    -> RankMixerBlock(mode="full")
    -> flatten
    -> MLP projection
    -> decode_residual
    -> alpha * gate * decode_residual
```

因此它和主干第一层 block 内部原有的：

```text
decoded_qs + ns_tokens -> RankMixerBlock -> next_q_list
```

是两条并行路径。新增分支复用第一层 raw `decoded_qs` 作为输入，但会在 shortcut branch 内独立演变后，再以 gated residual 的方式补到 classifier 前表示。

最终融合顺序固定为：

```text
HyFormer final Q output
    -> optional User-Item Shortcut Residual
    -> optional First Decoded-Q Residual
    -> Classifier
```

这样保留已有 `ui_shortcut` 行为，再在其后补充第一层 decoded-q 证据。

### 新增参数

训练入口新增：

```bash
--first_decoded_shortcut_type none|mixer|ffn_only|mlp
--first_decoded_detach
--first_decoded_alpha_init 0.01
```

含义如下：

- `none`：默认值，关闭分支，baseline 行为和参数结构保持不变
- `mixer`：对第一层 `decoded_qs` 使用 `RankMixerBlock(mode="full")`，再投影为 residual
- `ffn_only`：使用 `RankMixerBlock(mode="ffn_only")`，不做 token mixing
- `mlp`：直接 flatten `decoded_qs` 后走 MLP
- `first_decoded_detach`：detach 第一层 `decoded_qs`，让分支不把额外梯度传回第一层 HyFormerBlock
- `first_decoded_alpha_init`：可学习 residual scale 的初始值，默认 `0.01`

如果 `mixer` 模式下 `d_model % (num_queries * num_sequences) != 0`，会沿用 `RankMixerBlock` 的清晰报错，不自动 fallback。

### 模型侧改动

- `model.py` / `evaluation/model.py`
  - 新增 `FirstDecodedQResidualBranch`
  - `MultiSeqHyFormerBlock.forward()` 新增 `return_decoded_qs=False`
  - `_run_multi_seq_blocks()` 只在第一层捕获 block 内部 raw `decoded_qs`
  - 新增 `_apply_first_decoded_shortcut()`
  - `forward()` 和 `predict()` 都使用同一套融合逻辑
- `train.py`
  - 新增三个训练参数，并写入 `model_args`
  - `trainer.py` 保存的 `train_config.json` 会自然记录这些参数
- `evaluation/infer.py`
  - fallback 配置新增：

```python
'first_decoded_shortcut_type': 'none',
'first_decoded_detach': False,
'first_decoded_alpha_init': 0.01,
```

因此旧 checkpoint 在缺少这些字段时会默认按关闭分支构建模型。

### run.sh

训练参数里的 `first_decoded_alpha_init` 默认值仍然是 `0.01`，fallback 配置也使用 `0.01`。但当前 `run.sh` 默认实验显式启用了该分支，并将 `alpha` 设为 `0.2`：

```bash
--first_decoded_shortcut_type mixer \
--first_decoded_alpha_init 0.2
```

如果需要恢复原始 baseline 行为，可以显式覆盖：

```bash
bash run.sh --first_decoded_shortcut_type none
```

### 推荐实验命令

```bash
# 原始行为，必须兼容
bash run.sh --first_decoded_shortcut_type none

# 启用第一层 decoded-q residual 分支
bash run.sh \
  --first_decoded_shortcut_type mixer \
  --first_decoded_alpha_init 0.2

# 不做 token mixing，只做 FFN 消融
bash run.sh \
  --first_decoded_shortcut_type ffn_only \
  --first_decoded_alpha_init 0.2

# 只做 MLP 分支
bash run.sh \
  --first_decoded_shortcut_type mlp \
  --first_decoded_alpha_init 0.2

# detach 消融
bash run.sh \
  --first_decoded_shortcut_type mixer \
  --first_decoded_detach \
  --first_decoded_alpha_init 0.2
```


## 8. Target Attention / DIN

### 解决了什么问题？

当前主干虽然已经有：

- `ui_aware` query generator
- `user-item shortcut residual`
- `first decoded-q residual`

但它们的核心仍然更偏向于：

```text
先得到一个更好的最终表示，再用于分类
```

对于 CVR 任务来说，更直接的问题通常是：

```text
当前 item 到底应该激活用户历史里的哪些行为证据？
```

也就是说，除了让主干自己去隐式学习这件事之外，还可以显式增加一条：

```text
current item -> target attention over histories
```

这次新增的 `Target Attention / DIN` 分支就是在解决这个问题。它不替换现有 HyFormer 主干，也不替换 `ui_aware` query generator，而是在分类头前补一条 DIN-style 的目标相关历史聚合 residual。

### 新增了什么？

训练入口新增参数：

```bash
--target_attn_type none|din
--target_attn_alpha_init 0.1
```

含义如下：

- `target_attn_type=none`：默认基线行为，关闭该分支
- `target_attn_type=din`：启用 DIN-style target attention
- `target_attn_alpha_init`：该分支可学习 residual scale 的初始值，默认 `0.1`

当前 `run.sh` 已默认显式启用：

```bash
--target_attn_type din
```

因此当前默认实验配置里，Target Attention / DIN 是开启状态。

### DIN 分支现在是如何工作的？

当前实现没有直接拿 raw item id 去做 DIN，而是复用 tokenizer 之后的 item token 表示。整体流程可以概括为：

```text
item_tokens
    -> mean pooling
    -> item_repr

item_repr 对每个 sequence domain 的每个历史 token 打分：
score = MLP([q, h, q*h, q-h])

softmax over valid history positions
    -> domain context

每个 domain 的 context + domain_emb
    -> concat over all domains
    -> out_proj
    -> target_ctx

final_output = q_output + alpha * gate(q_output, target_ctx) * target_ctx
```

这里有几个实现细节需要说明：

- `q` 来自 `item_tokens.mean(dim=1)` 之后的 `item_repr`
- 每个 domain 独立计算 attention，不共享历史 softmax
- padding 位会先被 mask，再做归一化，避免全 padding 时数值异常
- 每个 domain 都会额外加一个 `domain_emb`，保留 `seq_a / seq_b / seq_c / seq_d` 的域区分信息
- 所有 domain context 拼接后，再投影回 1 个 `d_model` 向量作为 `target_ctx`
- 最终不是直接相加，而是经过 `gate + learnable alpha` 的 gated residual 融合

因此这条分支更像是：

```text
用当前 item 显式去用户历史中检索“与这个目标最相关的证据”
```

而不是再做一遍通用的 user/item matching。

### 和现有几个分支的关系

这条分支和已有模块的职责边界可以这样理解：

- `ui_aware query generator`：决定“当前 item 视角下，query 应该怎么生成，再去序列里检索”
- `ui_shortcut residual`：在分类头前补充浅层 user-item 非序列匹配信号
- `first decoded-q residual`：把第一层 query decode 得到的早期检索证据旁路送到分类头前
- `target_attn / DIN`：让当前 item 直接对各路历史序列做显式目标相关聚合

当前融合顺序是：

```text
HyFormer final Q output
    -> optional User-Item Shortcut Residual
    -> optional First Decoded-Q Residual
    -> optional Target Attention / DIN Residual
    -> Classifier
```

也就是说，DIN 分支是在主干最终表示已经形成之后，再补一条目标相关历史证据。

### 模型侧改动

- `model.py` / `evaluation/model.py`
  - 新增 `DINTargetAttention`
  - `PCVRHyFormer` 新增 `target_attn_type`
  - 新增 `_apply_target_attention()`
  - `forward()` / `predict()` 都会在 classifier 前走同一套 target attention 融合逻辑
- `train.py`
  - 新增 `--target_attn_type`
  - 新增 `--target_attn_alpha_init`
  - 将这两个参数写入 `model_args`
- `evaluation/infer.py`
  - fallback 配置新增：

```python
'target_attn_type': 'none',
'target_attn_alpha_init': 0.1,
```

因此如果 checkpoint 中带有 `train_config.json`，推理侧会按训练时保存的 target attention 配置恢复模型结构。

### 常用运行方式

关闭该分支，回到没有 DIN 的对照：

```bash
bash run.sh --target_attn_type none
```

启用当前默认的 DIN 分支：

```bash
bash run.sh --target_attn_type din
```

如果想手动调大或调小这条 residual 的初始强度：

```bash
bash run.sh \
  --target_attn_type din \
  --target_attn_alpha_init 0.1
```

也可以和现有 `ui_aware` / `ui_shortcut` 一起组合实验：

```bash
bash run.sh \
  --query_generator_type ui_aware \
  --ui_shortcut_type bi_cross_attn \
  --target_attn_type din
```


## 9. 扩大序列截断上限

### 解决了什么问题？

原有序列截断上限比较保守（`seq_a:256, seq_b:256, seq_c:512, seq_d:512`）。对于 CVR 任务来说，更长远的用户行为历史可能包含有价值的兴趣信号，过短的截断可能导致模型无法看到足够远的历史行为。

这次改动的目标是：

- 给各序列 domain 更充足的截断上限
- 对不同 domain 做差异化处理（`seq_d` 通常是增长最快的序列，给两倍空间）
- 通过 CLI 参数暴露，方便做消融实验

### 新增了什么？

训练入口已有参数：

```bash
--seq_max_lens "domain:max_len,domain:max_len,..."
```

当前 `run.sh` 默认配置已显式扩大：

```bash
--seq_max_lens "seq_a:512,seq_b:512,seq_c:512,seq_d:1024"
```

其中 `seq_d` 是 `seq_a/b/c` 的两倍，因为 `seq_d` 通常是用户会话级行为序列，长度增长最快。

### 显存影响

扩大截断上限会直接增大序列 buffer 的预分配尺寸，因此显存占用会增加。如果遇到 OOM，可以回退：

```bash
bash run.sh --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
```

### 推理侧兼容性

`evaluation/infer.py` 会从 `train_config.json` 读取 `seq_max_lens`，构造与训练时相同截断长度的数据集。如果 `train_config.json` 中缺少该字段，会回退到 `_FALLBACK_SEQ_MAX_LENS`。

### 推荐实验命令

```bash
# 当前默认（扩大后）
bash run.sh

# 回退到原始截断上限做对照
bash run.sh --seq_max_lens "seq_a:256,seq_b:256,seq_c:512,seq_d:512"

# 进一步扩大做消融
bash run.sh --seq_max_lens "seq_a:512,seq_b:512,seq_c:1024,seq_d:2048"
```

## 10. 序列时间统计特征（seq_time_stats）

### 解决了什么问题？

当前模型已经通过 `time_bucket` 将历史行为的距离信息引入了序列 token，但 `time_bucket` 是逐 token 粒度的离散分桶，没有直接刻画序列整体的活跃程度和近期行为密度。

对于 CVR 任务来说，用户在近期内的活跃程度（例如最近 1 小时内看过多少个商品、最近 7 天活跃频率等）本身就是重要的预测信号。另外，单纯依赖 `time_bucket` 无法捕捉跨越不同时间窗口的聚合行为模式。

这次新增的 `seq_time_stats` 就是在解决这个问题：为每个序列 domain 补充一组轻量级的时间窗口统计特征。

### 现在是如何工作的？

数据侧 `_derive_seq_time_stats()` 函数会基于当前样本的 `timestamp` 和各序列 `ts_padded`，对每个 domain 独立计算出 6 维统计量：

```text
stats[0] = log1p(valid_count)      # 有效历史行为总数（取对数压缩）
stats[1] = count(≤ 1h) / N         # 最近 1 小时内活跃比例
stats[2] = count(≤ 6h) / N         # 最近 6 小时内活跃比例
stats[3] = count(≤ 24h) / N        # 最近 1 天内活跃比例
stats[4] = count(≤ 7d) / N         # 最近 7 天内活跃比例
stats[5] = count(≤ 30d) / N        # 最近 30 天内活跃比例
```

其中 N 是该样本在当前 domain 的有效历史序列长度，用于归一化（避免活跃比例被序列截断长度带偏）。

### 模型侧如何使用？

`seq_time_stats` 在进入 `UserItemAwareQueryGenerator` 时通过 gated fusion 注入 query 生成过程：

```text
time_stats_proj: Linear(6, d_model) -> LayerNorm -> SiLU
time_context = time_stats_proj(seq_time_stats)
gate = sigmoid(MLP(cat([q_base, time_context], dim=-1)))
q_base = q_base + gate * time_context
```

也就是说，统计特征不会直接替代原有的 query base，而是以 gated residual 的方式补充，让模型自己决定"序列活跃程度应该在 query 生成中占多大权重"。

### 和现有时间特征的关系

当前模型内与时间相关的特征已经有四层：

- `time_bucket`：逐 token 粒度的距离分桶，影响序列 embedding
- `weekday/hour`：样本级日历特征，影响 ns token
- `seq_time_stats`：序列级聚合统计，影响 query 生成（新增）
- `target_attn / DIN`：当前 item 对历史序列的目标相关打分（self-attention 粒度的时序建模）

这四者的定位是不同的：

| 特征 | 粒度 | 作用位置 | 捕捉的信息 |
| --- | --- | --- | --- |
| `time_bucket` | token 级 | 序列 embedding | 每个历史距离当前样本多久 |
| `weekday/hour` | 样本级 | ns token | 样本发生的绝对时间模式 |
| `seq_time_stats` | 序列级 | query 生成 | 用户在序列各 domain 的活跃与衰减模式 |
| `target_attn / DIN` | 序列级 | 分类头前 | 当前 item 对各历史时刻的关联权重 |

### 训练侧与推理侧是否一致？

数据同步已确认：

- 训练侧 `dataset.py`：在 `__getitem__` 中通过 `_derive_seq_time_stats()` 计算并传递
- 推理侧 `evaluation/dataset.py`：使用完全相同的 `_derive_seq_time_stats()` 函数
- 训练侧 `train.py` 新增 `--use_seq_time_stats` 和 `--no_seq_time_stats` 参数
- `train_config.json` 中记录了 `seq_time_stats_dim`，推理时 `evaluation/infer.py` 正确恢复

### 新增开关

```bash
# 默认开启（run.sh 已显式传入）
--use_seq_time_stats

# 关闭
--no_seq_time_stats
```

### 推荐实验命令

```bash
# 当前默认（开启 seq_time_stats）
bash run.sh

# 关闭 seq_time_stats，做对照
bash run.sh --no_seq_time_stats
```
