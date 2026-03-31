# 计划：降低 per-layer/per-head attention 存储压力（不直接修改两函数）

## 1. 目标与成功标准

**目标**

* 在不直接修改 `get_saved_per_layer_head_attention` 与 `get_attention_from_saved_per_layer_head_fast` 的前提下，重新划分它们在整体流水线中的“职责边界”，并通过新增“保存侧压缩/过滤 + 读取侧解码/分析”的接口显著降低落盘存储压力。

**成功标准**

* 默认流水线不再落盘保存 `full_attn (layers, heads, T, T)` 这种全量 4D 矩阵。

* 单样本 attention 文件体积相对当前 full-attn 方案显著下降（通常可下降一个数量级以上，取决于 T、层数、头数与过滤强度）。

* 读取侧仍能生成与现有 fast 评估逻辑一致的可视化/指标（空间一致性、token 过滤等），并保留旧函数用于对照与 debug。

## 2. 现状分析（基于仓库代码）

### 2.1 现有两函数做了什么

* `get_saved_per_layer_head_attention`：

  * 从 `output_ids["attentions"]` 组装出 `full_attn`，形状为 `(num_layers, num_heads, total_len, total_len)`。

  * 这是“最重”的对象：total\_len 随 prompt+generation 增长，存储与 I/O 压力都非常大。

  * 代码位置：[visual\_tools.py:L497-L544](file:///h:/BUAA/工业异常检测/cvpr_code/eval_ad_pro/util/visual_tools.py#L497-L544)

* `get_attention_from_saved_per_layer_head_fast`：

  * 从 `llm_attn_matrix`（即上面 full\_attn 或同结构矩阵）中**只切片使用**两块信息：

    * `vlm_attn = output_tokens -> vision_tokens`（用于生成热力图/异常图）

    * `prompt2text_attn_all = output_tokens -> prompt_text_tokens`（用于 token 过滤、权重/阈值等）

  * 并在读取侧执行 outlier/spike 检测：

    * `detect_single_extreme_values_in_vlm_attn`

    * `detect_attn_spike_by_share`

  * 代码位置：[visual\_tools.py:L547-L731](file:///h:/BUAA/工业异常检测/cvpr_code/eval_ad_pro/util/visual_tools.py#L547-L731)

### 2.2 关键矛盾

* **保存侧构建/落盘的是 full 4D attention**，但读取侧只需要其中的“窄切片”（output->vision、output->prompt\_text）。

* outlier/spike 筛选逻辑在读取侧做，导致保存侧不得不把“坏行/坏token”的 attention 也落盘保存——进一步放大了无效存储。

### 2.3 已有可复用构件

* outlier/spike 检测函数在同文件顶部已定义：

  * [visual\_tools.py:L26-L44](file:///h:/BUAA/工业异常检测/cvpr_code/eval_ad_pro/util/visual_tools.py#L26-L44)

  * [visual\_tools.py:L46-L81](file:///h:/BUAA/工业异常检测/cvpr_code/eval_ad_pro/util/visual_tools.py#L46-L81)

* 仓库里已经存在“压缩存储 + 读取”的雏形（`optimized_get_*`），说明方向是可行的：

  * [visual\_tools.py:L734-L1040](file:///h:/BUAA/工业异常检测/cvpr_code/eval_ad_pro/util/visual_tools.py#L734-L1040)

## 3. 职责重划分（建议的分工）

> 这里的“安排”指：在流水线里让两函数各自承担什么角色；并通过新增 wrapper/新接口改变默认调用路径，而不是修改两函数本体。

### 3.1 `get_saved_per_layer_head_attention` 的建议职责（Debug/Fidelity 路径）

**定位**

* 保留为“全量重建 / 对照 / 调试”能力：当你需要分析某个样本的完整 attention 行为（例如论文可视化、错误溯源、对比不同压缩策略）时使用。

**不再承担**

* 不再作为“默认落盘格式”的来源函数（避免保存 full\_attn）。

* 不再承担任何 token/outlier 过滤职责（保持其纯粹性与可解释性）。

### 3.2 `get_attention_from_saved_per_layer_head_fast` 的建议职责（Legacy 读取路径）

**定位**

* 保留为“兼容旧数据/旧格式”的读取与分析函数：输入仍为 full\_attn 或同结构矩阵时使用。

**不再承担（在默认新链路里）**

* 不再承担 outlier/spike 检测（这部分前移到保存侧的压缩逻辑中）。

* 不再要求落盘数据一定是 full\_attn；默认新链路将走“压缩格式 + 新 reader wrapper”（见 3.3）。

### 3.3 新增一条默认链路：保存侧压缩 + 读取侧分析

**保存侧（新增函数/脚本承担）**

* 输入：`output_ids["attentions"]`、`sequences`、`input_token_len`、图像/patch 配置等。

* 只抽取两块必要 attention：

  * `attn_to_vision`: `(layers, heads, output_len, num_patches)`

  * `attn_to_text`: `(layers, heads, output_len, prompt_text_len)`

* 将 `(layers, heads, output_len, num_patches)` 展平到 token 维度：`vlm_attn = (layers*heads*output_len, num_patches)`。

* 对 `vlm_attn` 执行 outlier/spike 检测并删行：

  * `detect_single_extreme_values_in_vlm_attn(vlm_attn, ratio, dominance_ratio)`

  * `detect_attn_spike_by_share(vlm_attn, spike_patch_idx, share_thr)`

* 输出并落盘压缩结构：

  * `compressed_attn = {"vlm_attn", "prompt2text_attn", "kept_indices", "vlm_attn_normalized", "outlier_tokens_num", "all_tokens_num"}`

  * `meta = {input/output token 边界、vision token 边界、num_patches、layers/heads、patch/merge、vision_token_id}`

* 关键：落盘数据规模从 `O(L*H*T*T)` 下降为 `O(L*H*O*(P + Txt))`，且可进一步因删行下降。

**读取侧（新增 reader wrapper 承担）**

* 输入：`compressed_attn`、`sequences`、`meta`、图像与 tokenizer。

* 只做“后半段分析”：token 词性过滤、阈值/权重、空间熵、空间一致性、可视化输出等。

* 使用 `kept_indices` 把“压缩后 token 行”映射回原始 `layers*heads*output_len` 坐标系，保证输出 token 对齐与可解释性。

## 4. 接口与落盘格式（决策项）

### 4.1 建议落盘文件结构（单样本）

用 `torch.save` 保存一个 dict（`.pt`/`.pth`）：

* `compressed_attn`: dict（见 3.3）

* `sequence`: `sequences`（用于后续 tokenize/定位 vision token span）

* `meta`: dict（含 image\_path/question/output\_text 等业务字段 + attention 边界字段）

### 4.2 dtype / 压缩策略（建议）

* `vlm_attn`、`prompt2text_attn` 默认保存为 `float16` 或 `bfloat16`（视你的后处理容忍度）。

* 若对精度敏感：保存为 `float32`，但建议仍通过删行与切片大幅降体积。

* 可选增强：对 `vlm_attn` 每行只存 top-k（稀疏化），但这会影响空间熵与一致性度量，需要额外验证，不建议作为第一步默认策略。

## 5. 实施步骤（不改两函数本体）

1. **新增“保存侧压缩”入口**

   * 新建 `compress_and_save_attention(...)`（或等价模块/脚本），直接从 `output_ids["attentions"]` 抽取两块必要 attention，并执行 outlier/spike 过滤与落盘。

   * 调用侧脚本替换为使用该入口，而不是先构建 full\_attn 再存盘。

2. **新增“读取侧分析”入口**

   * 新建 `analyze_from_compressed_attention(...)`：读取压缩结构，继续执行 fast 分析逻辑（不再做 outlier/spike）。

3. **保留旧路径**

   * `get_saved_per_layer_head_attention` + `get_attention_from_saved_per_layer_head_fast` 维持原样，作为 legacy/debug。

   * 新脚本使用新链路；旧脚本可保持不动，或仅作为兼容入口。

4. 在代码入口文件中，只对optimized\_generate\_outputs.py和optimized\_qwen3\_evaluate\_from\_saved\_perhead\_fast.py做出对应的修改，不要改动generate\_outputs.py和qwen3\_evaluate\_from\_saved\_perhead\_fast.py

## 6. 风险点与边界情况

* **vision\_token\_id 定位失败**：保存侧应在 meta 中记录并在读取侧兜底（直接返回全局 attention 或跳过该样本）。

* **过滤过强导致 keep\_mask 为空**：保存侧应兜底保留全部行或保留最小数量行，避免评估崩溃。

* **prompt\_text\_len 为 0**：读取侧 token 过滤无意义，应直接回退到只用 `vlm_attn` 生成全局热力图。

* **模型层数/头数变化**：必须写入 meta，读取侧优先使用 meta 的 layers/heads。

## 7. 验证方案

### 7.1 存储压力验证

* 对同一批样本分别：

  * 旧链路：保存 full\_attn

  * 新链路：保存 compressed\_attn

* 统计单样本文件大小分布（均值/中位数/最大值），并记录节省比例。

### 7.2 结果一致性验证

* 选取少量样本：

  * 用旧链路得到最终异常图/可视化与指标

  * 用新链路得到同样输出

* 检查：

  * 输出图整体形态与阈值行为是否一致

  * SC/空间熵等指标是否在合理范围内（允许轻微差异）

## 8. 明确假设（当前计划依赖）

* 读取侧真正需要的 attention 信息仅限于：

  * output token 到 vision patch 的 cross-attn

  * output token 到 prompt 文本 token 的 cross-attn

* outlier/spike 过滤在保存侧执行不会破坏后续分析逻辑的正确性（因为读取侧本来也会剔除这些 token 行）。

