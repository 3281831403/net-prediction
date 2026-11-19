# net-prediction

## 题面

弱网下开预测回滚后，角色开始瞬移：预测按到达顺序而非输入 seq 应用，乱序包让画面抖动；回滚只重置根节点，子实体仍停在错误位置；历史窗口按时间裁剪，卡顿时丢掉关键帧；校验和对字段插入顺序敏感，同一状态得到不同哈希；插值 alpha 未限制在 0..1，负值把角色外推到墙外；去重只比较相邻 seq，非相邻重复包被重复应用；时钟偏移取均值，单个离群样本把偏移拉偏；校正量没有上限，一次瞬移穿过墙体；重放时把已确认输入又应用一遍，位置持续累积偏移；批量 reconcile 中途异常后历史与状态不一致。netcode/prediction.py 是核心，tests/test_prediction.py 15个红。要修的是：按 seq 稳定排序并校验帧号单调；回滚全部实体后只重放未确认输入；按帧数保留历史窗口；使用与字段插入顺序无关的稳定哈希；把 alpha 限制在 0..1 并越界抛 NetError；按 seq 全局去重并丢弃 <= last_seq 的输入；用中位数和 MAD 剔除离群样本；按 max_step 分帧限幅校正；只重放 tick > ack_tick 的输入；让 reconcile 原子化，异常时不改动历史与状态。

## 运行方式

在仓库根目录执行：

```
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，且没有 collection error、import error 或 syntax error。不要 sleep、不要联网，也不要依赖随机数或真实时钟；所有帧号和 `now` 都由参数显式传入。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 其他 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/` 下的测试不可修改。实现只使用 Python 标准库（`collections`、`dataclasses`、`math` 等）和 pytest；离线运行，不 `git init`、不 push、不联网。
