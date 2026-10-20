# Codex 修改文档：在原始 SCD-SAM 中融入 SSCLNet 风格语义伪标签生成模块




CUDA_VISIBLE_DEVICES=1 python train.py  --batch-size 2  --effective-batch-size 8 --epochs 100   --learning-rate 0.00035   --amp   --gradient-checkpointing   --no-use-ema-teacher  --no-use-semantic-pseudo  --seed 42
## 一、修改目标

请在原始 SCD-SAM 项目基础上新增一个半监督伪标签训练模块，参考 SSCLNet 的 self-training 思路：

1. 保留原始 SCD-SAM 的双时相输入结构；
2. 保留原始输出：

   * `outputs_A`：T1 / A 时相语义预测；
   * `outputs_B`：T2 / B 时相语义预测；
   * `out_change`：变化检测预测；
3. 新增 EMA Teacher，用 Teacher 为未标注 / 未变化区域生成语义伪标签；
4. 使用 entropy 熵筛选高质量伪标签；
5. 只在可靠伪标签位置计算 semantic pseudo loss；
6. 原始 change loss 仍然作为主损失；
7. 伪标签 loss 权重默认较小，避免 semantic pseudo 破坏原模型训练稳定性。

本次修改不引入 graph 模块，不改变 SCD-SAM 主体网络结构，只在训练流程中新增 Teacher–Student 伪标签训练逻辑。

---

## 二、总体训练结构

原始 SCD-SAM 训练流程大致为：

```text
imgs_A, imgs_B
    ↓
Student SCD-SAM
    ↓
outputs_A, outputs_B, out_change
    ↓
loss_sem_sup + loss_change
```

修改后训练流程为：

```text
imgs_A, imgs_B
    ↓
Student SCD-SAM
    ↓
outputs_A, outputs_B, out_change
    ↓
supervised semantic loss + change loss

imgs_A, imgs_B
    ↓
EMA Teacher SCD-SAM
    ↓
teacher_outputs_A, teacher_outputs_B, teacher_change
    ↓
candidate unchanged mask
    ↓
entropy class-wise filtering
    ↓
reliable pseudo labels
    ↓
semantic pseudo loss
```

最终总损失：

```python
loss = loss_change + lambda_sem_sup * loss_sem_sup + lambda_sem_u * loss_sem_u
```

默认建议：

```python
lambda_sem_sup = 1.0
lambda_sem_u = 0.01
```

如果训练稳定，再尝试调到：

```python
lambda_sem_u = 0.05
```

---

## 三、新增配置参数

请在训练参数中新增以下配置，保证默认关闭或保守启用，避免影响原始训练结果。

```python
--use_ema_teacher          # 是否启用 EMA Teacher，默认 True
--use_semantic_pseudo      # 是否启用 semantic pseudo label，默认 False 或 True 但 start epoch 较晚
--ema_decay                # EMA 更新系数，默认 0.99 或 0.999

--pseudo_start_epoch       # 开始使用伪标签的 epoch，默认 10 或 15
--pseudo_alpha_start       # 初始伪标签保留比例，默认 0.5
--pseudo_alpha_end         # 后期伪标签保留比例，默认 0.8
--pseudo_min_ratio         # 每类最小保留比例，默认 0.05 或 0.1
--pseudo_ignore_index      # 伪标签忽略值，默认 255

--pseudo_change_thr        # teacher change 判断未变化区域阈值，默认 0.3
--lambda_sem_u             # 未标注语义伪标签损失权重，默认 0.01
--lambda_sem_sup           # 有标注语义损失权重，默认 1.0
--pseudo_rampup_epoch      # 伪标签 loss ramp-up 轮数，默认 5
```

要求：当 `--use_semantic_pseudo False` 时，训练结果应与原始 SCD-SAM 基本一致。

---

## 四、EMA Teacher 设计

在训练开始时复制 Student 模型作为 Teacher：

```python
import copy

teacher_model = copy.deepcopy(model)
teacher_model.eval()

for p in teacher_model.parameters():
    p.requires_grad = False
```

每次 Student 完成一次 `optimizer.step()` 后，更新 Teacher：

```python
@torch.no_grad()
def update_ema_teacher(student, teacher, decay=0.99):
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()

    for key in teacher_state.keys():
        if teacher_state[key].dtype.is_floating_point:
            teacher_state[key].mul_(decay).add_(student_state[key].detach(), alpha=1.0 - decay)
        else:
            teacher_state[key].copy_(student_state[key])
```

注意：

1. Teacher 不参与反向传播；
2. Teacher 只用于生成伪标签；
3. Teacher forward 时使用 `torch.no_grad()`；
4. Teacher 可设为 `eval()`，Student 保持 `train()`；
5. 每个 iteration 或每个 epoch 后都可以 EMA 更新，建议每个 iteration 更新。

---

## 五、伪标签生成区域设计

参考 SSCLNet，伪标签不应该在变化区域随意生成，而应主要作用于未变化 / 未标注区域。

原始 SCD 数据中通常有：

```text
labels_A > 0 或 labels_B > 0：变化区域，已有语义标签；
labels_A == 0 且 labels_B == 0：未变化或未标注区域，没有语义标签。
```

候选区域建议定义为：

```python
gt_unchanged_mask = (labels_A == 0) & (labels_B == 0)
```

为了更稳定，进一步结合 Teacher 的变化预测：

```python
teacher_change_prob = torch.sigmoid(teacher_change).squeeze(1)
teacher_unchanged_mask = teacher_change_prob < pseudo_change_thr
candidate_mask = gt_unchanged_mask & teacher_unchanged_mask
```

即只在以下区域生成 semantic pseudo label：

```text
1. 数据标签认为该位置不是变化区域；
2. Teacher change 分支也认为该位置大概率未变化；
3. Teacher semantic 分支预测熵较低。
```

---

## 六、entropy 熵筛选机制

Teacher 输出语义 logits：

```python
teacher_outputs_A: [B, C, H, W]
teacher_outputs_B: [B, C, H, W]
```

先计算 softmax：

```python
prob = torch.softmax(logits, dim=1)
```

再计算每个像素的 entropy：

```python
entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
```

得到 Teacher 预测类别：

```python
pred = torch.argmax(prob, dim=1)
```

熵越低，说明 Teacher 越确定，该像素伪标签越可靠。

---

## 七、按类别动态筛选伪标签

参考 SSCLNet，不使用全局统一阈值，而是对每个类别分别筛选。

对于每个类别 `c`：

```text
1. 找出 Teacher 预测为 c 的候选像素；
2. 取这些像素的 entropy；
3. 按 entropy 从低到高排序；
4. 只保留前 gamma_c 比例的像素；
5. 其他像素设为 ignore_index。
```

动态比例：

```python
base_ratio = alpha_start + (alpha_end - alpha_start) * current_iter / total_iter
gamma_c = base_ratio * class_precision[c]
gamma_c = clamp(gamma_c, min=pseudo_min_ratio, max=alpha_end)
```

其中：

```text
alpha_start = 0.5
alpha_end = 0.8
class_precision[c] 表示上一阶段 / 当前 batch 中第 c 类在有标签区域上的 precision。
```

如果暂时不实现 class precision，可以先使用：

```python
class_precision = torch.ones(num_classes).to(device)
```

更稳妥的初始值：

```python
class_precision = torch.ones(num_classes).to(device) * 0.8
```

后续再加入 EMA precision 更新。

---

## 八、建议新增函数：generate_entropy_pseudo_label

请新增一个工具函数，例如放在：

```text
utils/pseudo_label.py
```

函数接口建议如下：

```python
def generate_entropy_pseudo_label(
    logits,
    candidate_mask,
    num_classes,
    gamma_per_class,
    ignore_index=255
):
    """
    Args:
        logits: Tensor, [B, C, H, W], teacher semantic logits
        candidate_mask: Bool Tensor, [B, H, W], where pseudo labels are allowed
        num_classes: int
        gamma_per_class: Tensor or list, length C, keep ratio for each class
        ignore_index: int

    Returns:
        pseudo_target: Long Tensor, [B, H, W], unreliable positions are ignore_index
        reliable_mask: Bool Tensor, [B, H, W]
        entropy: Tensor, [B, H, W]
    """
```

核心逻辑：

```python
import torch

@torch.no_grad()
def generate_entropy_pseudo_label(
    logits,
    candidate_mask,
    num_classes,
    gamma_per_class,
    ignore_index=255
):
    prob = torch.softmax(logits, dim=1)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
    pred = torch.argmax(prob, dim=1)

    pseudo_target = torch.full_like(pred, fill_value=ignore_index)
    reliable_mask = torch.zeros_like(pred, dtype=torch.bool)

    for cls in range(num_classes):
        cls_mask = (pred == cls) & candidate_mask

        num_valid = int(cls_mask.sum().item())
        if num_valid <= 0:
            continue

        ratio = float(gamma_per_class[cls])
        ratio = max(0.0, min(1.0, ratio))

        num_keep = int(num_valid * ratio)
        if num_keep <= 0:
            continue

        cls_entropy = entropy[cls_mask]

        if num_keep >= num_valid:
            threshold = cls_entropy.max()
        else:
            threshold = torch.topk(cls_entropy, k=num_keep, largest=False).values.max()

        keep_mask = cls_mask & (entropy <= threshold)

        pseudo_target[keep_mask] = pred[keep_mask]
        reliable_mask[keep_mask] = True

    return pseudo_target.long(), reliable_mask, entropy
```

注意：

1. 如果 SCD-SAM 的 supervised semantic loss 内部使用 `labels - 1` 的类别编码，请确保伪标签 target 与原始 CE target 编码一致；
2. 如果语义 head 输出类别数不包含 0 类，则 `pred` 是 `0 ~ C-1`，可以直接作为 CrossEntropy 的 target；
3. 如果原项目标签是 `1 ~ C`，而 loss 前需要转换，请复用原项目已有的转换逻辑，避免类别错位；
4. `ignore_index` 必须与 semantic CE loss 中的 `ignore_index` 一致。

---

## 九、训练循环修改方式

原始训练中 Student forward：

```python
outputs_A, outputs_B, out_change = model(imgs_A, imgs_B)
```

新增 Teacher forward：

```python
if use_ema_teacher and use_semantic_pseudo and epoch >= pseudo_start_epoch:
    with torch.no_grad():
        teacher_outputs_A, teacher_outputs_B, teacher_change = teacher_model(imgs_A, imgs_B)
```

构造候选区域：

```python
gt_unchanged_mask = (labels_A == 0) & (labels_B == 0)
teacher_change_prob = torch.sigmoid(teacher_change).squeeze(1)
teacher_unchanged_mask = teacher_change_prob < pseudo_change_thr
candidate_mask = gt_unchanged_mask & teacher_unchanged_mask
```

生成 A/B 时相伪标签：

```python
pseudo_A, reliable_A, entropy_A = generate_entropy_pseudo_label(
    logits=teacher_outputs_A,
    candidate_mask=candidate_mask,
    num_classes=num_classes,
    gamma_per_class=gamma_per_class,
    ignore_index=pseudo_ignore_index
)

pseudo_B, reliable_B, entropy_B = generate_entropy_pseudo_label(
    logits=teacher_outputs_B,
    candidate_mask=candidate_mask,
    num_classes=num_classes,
    gamma_per_class=gamma_per_class,
    ignore_index=pseudo_ignore_index
)
```

计算 semantic pseudo loss：

```python
loss_sem_u_A = criterion_sem(outputs_A, pseudo_A)
loss_sem_u_B = criterion_sem(outputs_B, pseudo_B)
loss_sem_u = loss_sem_u_A + loss_sem_u_B
```

如果当前 batch 中没有可靠伪标签：

```python
if reliable_A.sum() == 0 and reliable_B.sum() == 0:
    loss_sem_u = torch.tensor(0.0, device=outputs_A.device)
```

总损失：

```python
loss = loss_change + lambda_sem_sup * loss_sem_sup + lambda_sem_u_current * loss_sem_u
```

其中 `lambda_sem_u_current` 建议加入 ramp-up：

```python
if epoch < pseudo_start_epoch:
    lambda_sem_u_current = 0.0
else:
    ramp = min(1.0, (epoch - pseudo_start_epoch + 1) / pseudo_rampup_epoch)
    lambda_sem_u_current = lambda_sem_u * ramp
```

---

## 十、class precision 更新

可先实现简化版本：所有类别 precision 设为 1 或 0.8。

如果要实现 SSCLNet 风格动态 precision，请在有标签区域统计 Teacher 预测 precision。

示例：

```python
@torch.no_grad()
def update_class_precision(
    logits,
    labels,
    valid_mask,
    num_classes,
    old_precision=None,
    momentum=0.9,
    eps=1e-6
):
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

    precision = torch.ones(num_classes, device=logits.device)

    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid_mask
        tp = ((pred == cls) & (labels == cls) & valid_mask).sum().float()
        fp = ((pred == cls) & (labels != cls) & valid_mask).sum().float()

        if pred_cls.sum() > 0:
            precision[cls] = tp / (tp + fp + eps)
        else:
            precision[cls] = 1.0

    if old_precision is not None:
        precision = momentum * old_precision + (1.0 - momentum) * precision

    return precision.clamp(min=0.05, max=1.0)
```

注意：这里 `labels` 的编码必须和 `pred` 一致。如果原始标签是 `1 ~ C`，需要先转换为 `0 ~ C-1` 后再统计。

---

## 十一、有标签 semantic loss 保持原逻辑

不要破坏原始 SCD-SAM 的 supervised semantic loss。

如果原始 semantic label 中 0 表示无语义 / 未变化区域，则有标签语义损失只应在 `labels > 0` 区域计算：

```python
labels_A_sup = labels_A.clone()
labels_B_sup = labels_B.clone()

labels_A_sup[labels_A_sup == 0] = ignore_index
labels_B_sup[labels_B_sup == 0] = ignore_index
```

如果原项目已有处理方式，请优先复用原逻辑。

---

## 十二、change loss 保持主导

变化检测 loss 不要被 semantic pseudo loss 覆盖。

建议总损失中：

```python
loss_change 权重 = 1.0
lambda_sem_sup = 1.0
lambda_sem_u = 0.01
```

如果训练稳定，再试：

```python
lambda_sem_u = 0.05
```

不要一开始设置太大，例如不要直接设为 1.0。

---

## 十三、日志输出要求

请在训练日志中新增以下信息，方便判断伪标签质量：

```text
loss_change
loss_sem_sup
loss_sem_u
lambda_sem_u_current
pseudo_ratio_A
pseudo_ratio_B
candidate_ratio
entropy_A_mean
entropy_B_mean
gamma_mean
class_precision_mean
```

其中：

```python
pseudo_ratio_A = reliable_A.float().mean()
pseudo_ratio_B = reliable_B.float().mean()
candidate_ratio = candidate_mask.float().mean()
```

如果接入 TensorBoard，也请写入：

```text
train/loss_sem_u
train/pseudo_ratio_A
train/pseudo_ratio_B
train/candidate_ratio
train/lambda_sem_u
train/class_precision_mean
```

---

## 十四、验证指标保持原始输出

验证阶段不要使用 Teacher，也不要使用伪标签。

验证仍然使用 Student 模型输出：

```python
outputs_A, outputs_B, out_change = model(imgs_A, imgs_B)
```

继续计算原有指标：

```text
F1
IoU_change
Acc
mIoU
SeK
SCD score
```

新增伪标签模块后，至少需要比较：

```text
1. 原始 SCD-SAM baseline；
2. SCD-SAM + EMA Teacher，但不启用 pseudo loss；
3. SCD-SAM + semantic pseudo loss；
4. 不同 lambda_sem_u 的消融：0.01 / 0.05。
```

---

## 十五、稳定性保护

请加入以下保护逻辑：

1. 如果 `loss_sem_u` 为 NaN 或 Inf，则本 batch 跳过 semantic pseudo loss；
2. 如果 `candidate_mask.sum() == 0`，则 `loss_sem_u = 0`；
3. 如果可靠伪标签比例过低，例如 `< 0.001`，则打印 warning，但不中断训练；
4. 如果伪标签比例过高，例如 `> 0.8`，也打印 warning，说明筛选可能过宽；
5. 保证关闭 `use_semantic_pseudo` 时，原始训练完全可运行。

---

## 十六、推荐默认训练策略

建议默认阶段如下：

```text
epoch 0 ~ 9:
    只训练原始 SCD-SAM
    loss = loss_change + loss_sem_sup
    更新 EMA Teacher，但不使用 pseudo loss

epoch 10 ~ 14:
    启用 semantic pseudo
    lambda_sem_u 从 0 ramp-up 到 0.01

epoch 15 以后:
    正常使用 semantic pseudo
    lambda_sem_u = 0.01
```

如果 semantic mIoU 有提升且 change F1 不下降，再尝试：

```text
lambda_sem_u = 0.05
pseudo_change_thr = 0.3
pseudo_alpha_start = 0.5
pseudo_alpha_end = 0.8
```

---

## 十七、预期效果

本修改的目标不是直接提升 binary change F1，而是提升完整语义变化检测能力，尤其是：

```text
1. outputs_A / outputs_B 的语义预测更完整；
2. 未变化区域语义特征得到补充训练；
3. semantic mIoU、SCD mIoU、SeK 有机会提升；
4. change F1 尽量不下降。
```

因此评估时不要只看 change-only F1，还要重点查看：

```text
semantic mIoU
SCD mIoU
SeK
Fscd
```

---

## 十八、验收标准

完成修改后，请确保：

1. `--use_semantic_pseudo False` 时，训练流程与原始 SCD-SAM 一致；
2. `--use_semantic_pseudo True` 时，训练不报错；
3. Teacher 不参与反向传播；
4. EMA Teacher 每个 iteration 正常更新；
5. 伪标签只在未变化 / 未标注候选区域生成；
6. 高熵像素被设为 `ignore_index`；
7. semantic pseudo loss 只作用于可靠伪标签；
8. 日志中可以看到 `loss_sem_u`、`pseudo_ratio_A/B`、`candidate_ratio`；
9. 验证阶段只使用 Student，不使用 Teacher；
10. 最终 checkpoint 保存 Student 模型，同时可选保存 Teacher 模型。
