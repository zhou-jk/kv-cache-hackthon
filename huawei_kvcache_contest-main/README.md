# KV Cache 策略评测说明

本文档说明如何使用本仓库对 KV Cache 策略进行简单评测，并产出命中率等指标。

## 目录结构

- `kvcachepolicy.py`：缓存淘汰策略（当前实现为 FIFO，内部仅调用 store 的 `add/delete`）。
- `kvstore.py`：简易 KV 缓存存储抽象，支持容量控制与增删查（容量在此处设定并强制）。
- `test.py`：评测脚本，读取输入数据并计算命中率。
- `input_samples/`：示例输入数据。

## 输入数据格式

输入文件每行的含义如下：

1. 第 1 行：缓存容量（正整数）。
2. 第 2 行起：一条请求/访问记录，格式为：`{prefix_hash_id 列表} type`

示例（`input_samples/sample1`）：

```
3
{1,2,3,4} 1
{1,2,3,4,5} 1
{1,2,6} 2
```

说明：
- 以上示例中缓存容量为 3。
- 每条记录的花括号内是该请求的 `prefix_hash_id` 访问序列；空格后的最后一个整数是该请求的 `type`（在 FIFO 策略中忽略）。
- 评测时会按序逐个访问花括号内的 `prefix_hash_id`，对每次访问统计命中或未命中，并按 FIFO 规则进行插入/淘汰。

## 运行评测

- 使用默认示例：

```bash
python3 test.py
```

- 指定自定义输入文件：

```bash
python3 test.py path/to/your_input_file
```

## 输出说明

脚本会输出如下四个指标：

- `total`：总请求数
- `hits`：命中次数
- `misses`：未命中次数
- `hit_ratio`：命中率（`hits/total`）

示例输出：

```
total=12 hits=0 misses=12 hit_ratio=0.0000
```

## 请自行预处理一下样例，可用于测试

https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
