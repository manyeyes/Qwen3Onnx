# Qwen3 ONNX .NET 推理项目使用与集成指南

本文面向使用者、.NET 集成开发者和后续推理引擎扩展开发者，介绍项目架构、Qwen3 与当前 ONNX 模型约定、模型获取、构建运行、CLI、控制台和桌面端使用、生成参数、C# 调用 API、日志排查及新增模型引擎的方法。

> 代码行为可能随版本演进。命令行的最终参数列表请始终以 `Qwen3.Cli --help` 输出为准。

## 目录

1. [项目简介](#1-项目简介)
2. [Qwen3 模型简介](#2-qwen3-模型简介)
3. [架构与数据流](#3-架构与数据流)
4. [环境与依赖](#4-环境与依赖)
5. [下载模型与准备文件](#5-下载模型与准备文件)
6. [构建与运行](#6-构建与运行)
7. [CLI 操作说明](#7-cli-操作说明)
8. [GenerationConfig 参数参考](#8-generationconfig-参数参考)
9. [控制台交互说明](#9-控制台交互说明)
10. [核心引擎 C# 集成](#10-核心引擎-c-集成)
11. [扩展到其他模型](#11-扩展到其他模型)
12. [日志、错误和排障](#12-日志错误和排障)
13. [当前限制与后续扩展方向](#13-当前限制与后续扩展方向)
14. [Avalonia 桌面端](#14-avalonia-桌面端)

## 1. 项目简介

本项目将 Qwen3 本地 ONNX 推理封装为四个 .NET 工程：

| 工程 | 类型 | 职责 |
| --- | --- | --- |
| `Qwen3.Inference` | 类库 | 配置、推理引擎抽象与工厂、ONNX Runtime 实现、生成策略、提示词、对话历史及统计结果 |
| `Qwen3.Cli` | 控制台应用 | 单次问答命令行入口、参数解析与验证、帮助、日志与退出码 |
| `Qwen3.Console` | 控制台应用 | 面向人工交互的多轮聊天体验、场景和思考模式命令 |
| `Qwen3.Desktop` | Avalonia 桌面应用 | MVVM 聊天界面、模型管理、SQLite 历史和本地账户 |

解决方案文件为 `Qwen3Onnx.sln`。所有应用依赖 `Qwen3.Inference`，模型会在 assistant 创建时加载，在进程内复用；一次 CLI 进程处理一个请求，交互式 Console 进程可持续处理多轮输入。

## 2. Qwen3 模型简介

Qwen3 是阿里巴巴通义千问团队发布的大语言模型系列，包含不同参数规模和模型结构。本项目列出的 0.6B 与 1.7B ONNX 仓库是面向本地运行的小规模 checkpoint；本项目不负责模型训练、转换或量化。

Qwen3 的对话使用方式支持思考型与非思考型生成；本项目通过 ChatML 风格的 system/user/assistant 消息边界组织提示词，并通过 `<think>` 与 `</think>` 标记影响思考区段。具体能力、支持语言、训练方式、许可与建议提示词应以对应模型卡及上游 Qwen3 发布资料为准。

模型参数规模、量化方案会影响磁盘占用、运行时内存、速度与输出质量。项目列出的 ModelScope 仓库包括 0.6B / 1.7B 两个规模及 INT8 / W4A32 两种量化名称：

| 模型仓库 | 页面 |
| --- | --- |
| Qwen3-0.6B-int8-onnx | https://modelscope.cn/models/manyeyes/Qwen3-0.6B-int8-onnx |
| Qwen3-0.6B-w4a32-onnx | https://modelscope.cn/models/manyeyes/Qwen3-0.6B-w4a32-onnx |
| Qwen3-1.7B-int8-onnx | https://modelscope.cn/models/manyeyes/Qwen3-1.7B-int8-onnx |
| Qwen3-1.7B-w4a32-onnx | https://modelscope.cn/models/manyeyes/Qwen3-1.7B-w4a32-onnx |

模型页面的文件、目录结构、许可和更新状态由模型发布者维护。模型权重不包含在本代码仓库内，也不会由程序自动下载。使用前需自行确认模型许可证、ModelScope 文件清单及其依赖数据文件。

### 2.1 思考模式与项目控制

- `ThinkMode = true` 时，输入提示词以 assistant 消息开头，由模型按 checkpoint 行为决定是否输出思考内容。
- `ThinkMode = false` 时，当前引擎会在 assistant 前缀后附加一个空的 `<think>...</think>` 区段，尝试跳过思考阶段。
- `ThinkingBudget > 0` 时，如果生成步数达到预算且思考尚未结束，引擎会尝试注入 tokenizer 编码得到的 `</think>` token。
- `ShowThinking` 是输出层选项：CLI 的 `--hide-thinking` 会过滤显示内容，但不会改变模型是否执行思考。
- `StripThinkingInHistory` 决定写入多轮记忆时是否移除思考区段。

模型是否遵循思考标记仍依赖模型本身和其 tokenizer，参数并不保证模型内部推理行为或答案内容。

### 2.2 INT8 与 W4A32 的使用边界

INT8/W4A32 是模型仓库发布名称中的量化方案标识；应以对应仓库的模型卡和文件清单为准。当前 `OnnxInferenceEngine` 的默认文件名是 `embedding.int8.onnx` 和 `decoder.int8.onnx`，输入输出契约包含 hidden states、逐层 key/value cache、历史长度、当前 token 长度和 attention mask。

使用 W4A32 时必须确认：

1. 仓库提供的 embedding / decoder 图和所需 tokenizer、config 文件完整。
2. 输入输出名称、顺序、形状、类型和 KV Cache 行为符合当前引擎实现。
3. ONNX Runtime 1.23.2 支持模型图中的算子与量化表示。
4. 如果文件名不同，通过 C# `AssistantConfig` 设置 `EmbeddingModelName` 与 `DecoderModelName`；当前 CLI 尚无对应文件名参数。

不能仅凭模型仓库名称判断它一定能被当前引擎无修改加载。若图契约不同，应新增引擎实现，而不是只修改文件名。

## 3. 架构与数据流

```text
Qwen3.Cli ───────┐
                 ├── ChatAssistant ── IInferenceEngineFactory ── IInferenceEngine
Qwen3.Console ───┘                                         └── OnnxInferenceEngine
                                                              ├── Tokenizers.DotNet
                                                              └── ONNX Runtime
```

### 3.1 应用层

- `Qwen3.Cli` 解析单条请求；将选定场景作为基础配置，再应用显式命令行覆盖项。
- `Qwen3.Console` 管理交互命令、全局 think 覆盖和场景选择。
- 两者通过 `ChatAssistant` 使用统一的对话 API。

### 3.2 推理协调层

`ChatAssistant` 负责：

- 根据显式场景或问题内容选择 `GenerationConfig`。
- 将多轮模式、think 模式或 system prompt 的变化反映到会话状态。
- 拼接 system/user/history 提示词；通过引擎接口调用 tokenizer、prefill 和 generate。
- 按配置管理历史轮数、历史 token 限制、thinking 文本清理与复读答案检查。
- 返回 `AssistantResult`，包含回答、生成统计和当前请求信息。

### 3.3 模型引擎层

`IInferenceEngine` 把模型相关操作抽象出来，包括 token 计数、cache 重置、历史 prefill、system/history/user 提示词格式化和生成。当前工厂默认注册 `qwen3-onnx`，由 `InferenceEngineFactory` 创建 `OnnxInferenceEngine`。

当前 ONNX 引擎执行的主要步骤：

1. 从模型目录读取 `config.json` 与 `tokenizer.json`。
2. 创建 embedding 和 decoder `InferenceSession`。
3. 校验 decoder 的输入名和输入/输出数量。
4. 将提示词编码为 token ID，通过 embedding 图得到 hidden states。
5. 调用 decoder 图，并在每层保存 key/value cache。
6. 依据 `GenerationConfig` 选择下一个 token，直到 stop token、长度限制或循环检测触发。
7. 解码生成 token 并返回文本及性能统计。

## 4. 环境与依赖

### 4.1 开发环境

- .NET 8 SDK
- Windows、Linux 或 macOS（各平台需要对应的 .NET 8 Runtime）
- 能读取模型仓库所提供 ONNX 图的 CPU 环境

### 4.2 NuGet 依赖

| 组件 | 当前版本 | 用途 |
| --- | --- | --- |
| `System.CommandLine` | 2.0.13 | CLI 参数、帮助、解析与选项校验 |
| `Microsoft.ML.OnnxRuntime` | 1.23.2 | ONNX 模型执行 |
| `Tokenizers.DotNet` | 1.4.1 | 载入 Hugging Face `tokenizer.json` 并执行编码/解码 |
| `Tokenizers.DotNet.runtime.*` | 1.4.1 | Windows x64、Linux x64/arm64、macOS x64/arm64 Tokenizer 原生运行时 |
| `Microsoft.Extensions.Logging.Abstractions` | 9.0.0 | 推理类库日志抽象 |
| `Microsoft.Extensions.Logging.Console` | 9.0.0 | CLI 控制台日志 |
| `Avalonia` / `Avalonia.Desktop` / `Avalonia.Themes.Fluent` | 12.1.4 | 跨平台桌面 UI |
| `CommunityToolkit.Mvvm` | 8.4.2 | MVVM ObservableObject 与命令生成 |
| `Microsoft.Data.Sqlite` | 9.0.0 | 本地会话、账户、模型目录及用量数据库 |

当前引擎使用 ONNX Runtime CPU Execution Provider。GPU Provider 尚未由该实现配置。

## 5. 下载模型与准备文件

### 5.1 选择模型

按目标资源、延迟和质量需求选择 0.6B 或 1.7B，再从以下页面选择 INT8 或 W4A32：

- [Qwen3-0.6B-int8-onnx](https://modelscope.cn/models/manyeyes/Qwen3-0.6B-int8-onnx)
- [Qwen3-0.6B-w4a32-onnx](https://modelscope.cn/models/manyeyes/Qwen3-0.6B-w4a32-onnx)
- [Qwen3-1.7B-int8-onnx](https://modelscope.cn/models/manyeyes/Qwen3-1.7B-int8-onnx)
- [Qwen3-1.7B-w4a32-onnx](https://modelscope.cn/models/manyeyes/Qwen3-1.7B-w4a32-onnx)

在 ModelScope 页面下载仓库文件到本地。模型规模和量化方式对内存/吞吐/质量的影响取决于具体导出文件、硬件和 Runtime，不在此对资源占用作固定承诺。

### 5.2 目录约定

当前 `OnnxInferenceEngine` 默认读取：

```text
<模型目录>\
  tokenizer.json
  config.json
  embedding.int8.onnx
  decoder.int8.onnx
```

如果 ONNX 权重以外部数据文件形式存储，必须连同外部数据文件一起下载，并维持导出图要求的相对路径。外部数据文件一般应与 ONNX 图保持仓库原始相对位置。

模型目录名称可以自行决定，例如：

```text
D:\models\Qwen3-0.6B-int8-onnx\
```

但**目录名称不负责转换模型文件名**。0.6B/1.7B 通常可以通过不同目录选择权重；如果某个仓库的 graph 文件名不同，CLI 当前不能直接指定文件名，需要使用 C# API 修改配置，或在未来增加对应 CLI 选项。

### 5.3 模型文件校验

`Qwen3.Console` 启动时会检查 tokenizer、config 和默认 embedding/decoder 文件是否存在，并列出缺少的路径。CLI 会在构造 ONNX 引擎时报告缺失文件。文件存在不代表 ONNX 图兼容；引擎初始化还会校验模型参数及 decoder 输入输出契约。

## 6. 构建与运行

在解决方案根目录（包含 `Qwen3Onnx.sln`）执行：

```powershell
dotnet restore Qwen3Onnx.sln
dotnet build Qwen3Onnx.sln
```

### 6.1 通过环境变量配置模型目录

PowerShell 当前会话：

```powershell
$env:QWEN_MODEL_PATH = "D:\models\Qwen3-0.6B-int8-onnx"
```

清除当前会话变量：

```powershell
Remove-Item Env:QWEN_MODEL_PATH
```

### 6.2 发布 CLI

Windows x64 示例：

```powershell
dotnet publish Qwen3.Cli\Qwen3.Cli.csproj -c Release -r win-x64 --self-contained false
```

发布后的模型文件仍需单独部署。检查发布输出中是否包含本机运行所需的 ONNX Runtime 和 Tokenizers 原生依赖；不同发布方式/Runtime ID 下应在目标机器验证。

## 7. CLI 操作说明

### 7.1 基础用法

```text
Qwen3.Cli <question> [options]
```

查看帮助：

```powershell
dotnet run --project Qwen3.Cli -- --help
dotnet run --project Qwen3.Cli -- -h
dotnet run --project Qwen3.Cli -- -?
```

查询版本：

```powershell
dotnet run --project Qwen3.Cli -- --version
```

单次问答：

```powershell
dotnet run --project Qwen3.Cli -- "用简短语言介绍 Qwen3"
```

问题中含空格或特殊字符时请使用引号。PowerShell 中 `--` 用来将选项传给 `dotnet run` 所启动的 CLI。

### 7.2 指定场景和 think 模式

```powershell
dotnet run --project Qwen3.Cli -- "求解 12 * 13" --scene deep
dotnet run --project Qwen3.Cli -- "解释什么是 ONNX" --scene fast_qa --think-mode off
dotnet run --project Qwen3.Cli -- "分析这个设计的优缺点" --scene deep --think-mode on
```

- 不传 `--scene`：根据问题内容执行启发式场景检测。
- 传 `--scene`：使用指定场景配置作为基线。
- 不传 `--think-mode` 或指定 `auto`：使用当前场景默认的 `ThinkMode`。
- 指定 `on` 或 `off`：覆盖场景的 think 默认值。

可选场景：`fast_qa`、`deep`、`roleplay`、`precise`、`chitchat`。

### 7.3 覆盖生成参数

```powershell
dotnet run --project Qwen3.Cli -- "写一段简短介绍" `
  --scene chitchat `
  --max-new-tokens 256 `
  --temperature 0.6 `
  --top-p 0.95 `
  --top-k 20 `
  --repetition-penalty 1.05 `
  --thinking-budget 192
```

命令行只覆盖显式给出的选项；其余参数沿用场景默认值。

### 7.4 流式与非流式

```powershell
dotnet run --project Qwen3.Cli -- "你好" --stream
dotnet run --project Qwen3.Cli -- "你好" --no-stream
```

CLI 按场景默认值决定流式方式；命令行显式给出正反选项可覆盖默认值。流式输出写入标准输出，日志写入标准错误。

### 7.5 日志

```powershell
dotnet run --project Qwen3.Cli -- "测试问题" `
  --log-level debug `
  --log-file ".\logs\inference.jsonl"
```

- `--log-level` 支持 `trace`、`debug`、`information`、`warning`、`error`、`none`，默认 `information`。
- 日志输出到标准错误流，不会与 stdout 中的模型回答混在一起。
- `--log-file` 可选；日志以 JSON Lines 格式追加写入。父目录不存在时会尝试创建。
- 文件日志包含 UTC 时间、等级、类别、事件 ID、格式化消息和异常文本。
- 日志可能包含模型目录路径和异常诊断信息。共享日志前请检查其中是否存在不应公开的信息。

### 7.6 CLI 退出码

| 退出码 | 说明 |
| --- | --- |
| `0` | 请求成功，或显示帮助/版本 |
| `1` | 命令行解析错误、模型文件缺失或模型/推理运行错误 |
| `2` | 生成配置无效或所选引擎未注册 |

参数拼写错误、枚举值不合法、数值转换失败、范围错误和互斥选项冲突会给出 CLI 错误输出。

## 8. GenerationConfig 参数参考

### 8.1 CLI 参数完整列表

下表中的默认值取自当前 `GenerationConfig.SceneConfigs`。如果指定场景，未在 CLI 显式覆盖的项沿用该场景默认值，而不是始终采用通用 `GenerationConfig` 构造默认值。

`GenerationConfig` 本身的通用构造默认值如下；场景配置只覆盖其中一部分：

| 属性 | 通用默认值 |
| --- | --- |
| `ThinkMode` / `ShowThinking` | `true` / `true` |
| `ThinkingBudget` | `256` |
| `MultiTurn` | `true` |
| `MaxTurns` | `3` |
| `MaxHistoryChars` / `MaxHistoryTokens` | `400` / `800` |
| `MaxNewTokens` / `Stream` | `256` / `true` |
| `Temperature` / `TopP` / `TopK` | `0.6` / `0.95` / `20` |
| `RepetitionPenalty` / `DoSample` | `1.1` / `true` |
| `SystemPrompt` | `null`（使用 `AssistantConfig.SystemPrompt`） |
| `StripThinkingInHistory` / `SanitizeHistory` | `true` / `true` |
| `ResetOnFactQuestion` / `EnableRepetitionCheck` | `true` / `true` |
| `RepetitionThreshold` | `0.9` |
| `EnableLoopDetection` / `LoopCheckWindow` / `LoopRepeatThreshold` | `true` / `20` / `4` |

| CLI 参数 | 对应属性 | 含义 | 验证/约束 |
| --- | --- | --- | --- |
| `--max-new-tokens` | `MaxNewTokens` | 本次最多生成 token 数 | 必须大于 0 |
| `--temperature` | `Temperature` | 采样温度；0 表示不缩放 logits | 有限数且 >= 0 |
| `--top-p` | `TopP` | nucleus sampling 概率阈值 | 有限数，(0, 1] |
| `--top-k` | `TopK` | 只保留概率最高的 K 个候选；0 表示不启用 Top-K | >= 0 |
| `--repetition-penalty` | `RepetitionPenalty` | 对生成中已出现过的 token 惩罚 | 有限数且 > 0 |
| `--sample` / `--no-sample` | `DoSample` | 开启采样或使用贪心 argmax | 两项互斥 |
| `--thinking-budget` | `ThinkingBudget` | think 未关闭时允许的思考 token 步数 | >= 0；0 表示不按 budget 强制结束 |
| `--think-mode auto\|on\|off` | `ThinkMode` | 场景 think 默认值或显式开关 | 只接受列出的值 |
| `--show-thinking` / `--hide-thinking` | `ShowThinking` | 显示或过滤思考标签区间 | 两项互斥；只影响显示 |
| `--stream` / `--no-stream` | `Stream` | 逐 token 输出或生成完成后输出 | 两项互斥 |
| `--multi-turn` / `--single-turn` | `MultiTurn` | 启用/关闭会话历史 | 两项互斥 |
| `--max-turns` | `MaxTurns` | 历史中保留的最大问答轮数 | >= 0 |
| `--max-history-chars` | `MaxHistoryChars` | 每条 assistant 历史答案截断字符数 | >= 0；0 会将非空答案截断为 `...` |
| `--max-history-tokens` | `MaxHistoryTokens` | 历史 token 超过上限时丢弃最旧轮次 | >= 0 |
| `--strip-thinking-in-history` / `--keep-thinking-in-history` | `StripThinkingInHistory` | 存历史前移除或保留思考文本 | 两项互斥 |
| `--sanitize-history` / `--no-sanitize-history` | `SanitizeHistory` | 存历史时清理 roleplay 括号片段 | 两项互斥 |
| `--reset-on-fact-question` / `--no-reset-on-fact-question` | `ResetOnFactQuestion` | 判断为独立事实问题时清理既有历史 | 两项互斥 |
| `--repetition-check` / `--no-repetition-check` | `EnableRepetitionCheck` | 检查新答案是否与近期答案高度相似 | 两项互斥 |
| `--repetition-threshold` | `RepetitionThreshold` | 相似度超过阈值时判定重复 | 有限数，[0, 1] |
| `--loop-detection` / `--no-loop-detection` | `EnableLoopDetection` | 检查 token 序列循环并提前停止 | 两项互斥 |
| `--loop-check-window` | `LoopCheckWindow` | 检查最近多少个生成 token | > 0 |
| `--loop-repeat-threshold` | `LoopRepeatThreshold` | 重复模式触发循环停止的次数 | > 0 |
| `--system-prompt` | `SystemPrompt` | 请求使用的 system prompt | 任意文本 |

模型和运行相关的 CLI 选项：

| 参数 | 含义 |
| --- | --- |
| `<question>` | 必需的单次用户问题 |
| `--model-path` | 模型目录；默认读取 `QWEN_MODEL_PATH`，再回退到 `AppContext.BaseDirectory\models\Qwen3-1.7b-int8-onnx` |
| `--engine` | 引擎注册 ID；默认 `qwen3-onnx` |
| `--scene` | 场景名；缺省时自动检测 |
| `--log-level` | 最低日志等级；默认 `information` |
| `--log-file` | 追加 JSON Lines 日志的路径 |
| `--help`、`-h`、`-?` | 显示帮助 |
| `--version` | 显示 CLI 程序版本 |

### 8.2 预定义场景默认值

| 场景 | Think | Think budget | 多轮 | Max turns | History chars | History tokens | Max new tokens | Temperature | Top-P | Top-K | Penalty | Sampling |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fast_qa` | 关 | 0 | 关 | 3 | 400 | 800 | 192 | 0.2 | 0.9 | 20 | 1.05 | 开 |
| `deep` | 开 | 384 | 关 | 3 | 400 | 800 | 768 | 0.4 | 0.95 | 20 | 1.05 | 开 |
| `roleplay` | 开 | 256 | 开 | 5 | 600 | 1536 | 384 | 0.7 | 0.95 | 40 | 1.1 | 开 |
| `precise` | 关 | 0 | 关 | 3 | 400 | 800 | 512 | 0.15 | 0.9 | 10 | 1.0 | 关 |
| `chitchat` | 开 | 192 | 开 | 3 | 400 | 1024 | 256 | 0.6 | 0.95 | 20 | 1.05 | 开 |

上表未逐项列出但仍从通用配置继承的属性包括 `ShowThinking`、`EnableRepetitionCheck`、`RepetitionThreshold`、`EnableLoopDetection`、`LoopCheckWindow`、`LoopRepeatThreshold`、`StripThinkingInHistory` 和 `SanitizeHistory` 等。

注意：`--max-turns` 的默认值对单轮场景仍有显示值，但 `MultiTurn = false` 时不会写入历史。`precise` 关闭采样，所以温度、Top-P、Top-K 对 token 选择不产生通常的采样效果。

### 8.3 场景检测规则

未指定场景时，推理核心使用简单关键词/正则规则：

1. 包含角色扮演提示（例如“请你扮演”） -> `roleplay`
2. 包含精确复述提示（例如“全文”“一字不差”） -> `precise`
3. 包含数学/证明提示（例如“计算”“求解”或数字运算符表达式） -> `deep`
4. 问题长度不超过 15 个字符且匹配部分事实型问句 -> `fast_qa`
5. 其他情况 -> `chitchat`

这不是语义分类器；场景分类不符合需要时，应显式传入 `--scene`。

## 9. 控制台交互说明

运行：

```powershell
dotnet run --project Qwen3.Console -- "D:\models\Qwen3-0.6B-int8-onnx"
```

或者先设置 `QWEN_MODEL_PATH` 再运行：

```powershell
$env:QWEN_MODEL_PATH = "D:\models\Qwen3-0.6B-int8-onnx"
dotnet run --project Qwen3.Console
```

### 9.1 内置命令

| 输入 | 操作 |
| --- | --- |
| `exit`、`quit`、`q` | 退出 |
| `clear` | 清屏 |
| `reset` | 清空对话历史和 KV Cache |
| `memory` | 显示当前历史轮数及内容摘要 |
| `scenes` | 显示场景及主要生成参数 |
| `config` | 显示模型目录、序列上限、线程数、system prompt 和 think 状态 |

### 9.2 思考模式

单次问题前缀：

```text
think:on 请分步骤分析这个问题
think:off 请直接给出结论
think:auto 恢复本问题所选场景的默认 think 模式
```

全局开关：

```text
/think on
/think off
/think auto
```

切换全局 think 模式时会清空历史和 KV Cache。单次前缀的优先级高于全局设置；`auto` 将此次请求交还给场景默认值。

### 9.3 指定场景

```text
scene:fast_qa Qwen3 是什么？
scene:deep 求解 12 * 13
scene:roleplay 请你扮演一位科普讲解员
```

### 9.4 当前 Console 与 CLI 的职责差别

`Qwen3.Console` 是多轮交互应用，场景使用预设 `GenerationConfig`，主要提供场景前缀和 think 控制。`Qwen3.Cli` 是单次命令行应用，提供本指南第 8 节列出的生成选项。需要对多轮生成细节逐项配置时，应在 C# API 设置 `GenerationConfig`，或使用 CLI 单次请求。

## 10. 核心引擎 C# 集成

### 10.1 引用类库

在应用项目中添加对 `Qwen3.Inference` 的项目或 NuGet 引用。若直接引用源码工程：

```powershell
dotnet add .\MyApp\MyApp.csproj reference .\Qwen3.Inference\Qwen3.Inference.csproj
```

部署时还需包含目标平台对应的 Tokenizers 原生运行时，以及 ONNX Runtime 依赖。

### 10.2 最小单次问答

```csharp
using Qwen3.Inference;

var config = new AssistantConfig
{
    EngineId = "qwen3-onnx",
    ModelPath = @"D:\models\Qwen3-0.6B-int8-onnx",
    SystemPrompt = "You are a helpful assistant.",
    MaxSequenceLength = 4096,
    IntraOpNumThreads = 4,
    InterOpNumThreads = 1
};

using var assistant = new ChatAssistant(config);
var result = assistant.Ask("请用中文简要介绍 Qwen3");

Console.WriteLine(result.Answer);
Console.WriteLine($"Tokens: {result.Generation.TotalTokens}");
Console.WriteLine($"Tokens/s: {result.Generation.TokensPerSecond:F2}");
```

`ChatAssistant` 创建时会通过工厂初始化引擎和模型。应复用同一个 assistant 处理需要共享历史的连续请求，并在结束时释放。

### 10.3 指定场景和覆盖参数

```csharp
var generation = GenerationConfig.SceneConfigs[Scene.DeepReasoning] with
{
    MaxNewTokens = 512,
    Temperature = 0.4,
    TopP = 0.95,
    TopK = 20,
    RepetitionPenalty = 1.05,
    ThinkingBudget = 256,
    MultiTurn = false,
    Stream = true
};

var result = assistant.Ask(
    "解释自回归解码中的 KV Cache",
    scene: Scene.DeepReasoning,
    generationConfig: generation,
    onToken: token => Console.Write(token));

Console.WriteLine();
Console.WriteLine($"场景: {result.SceneName}");
Console.WriteLine($"Prompt tokens: {result.PromptTokens}");
Console.WriteLine($"KV/cache 当前长度: {result.Generation.TotalTokens} 个新生成 token");
```

使用 `with` 从场景配置派生，可避免重建一整套默认值。若不指定 `generationConfig`，`scene` 对应预设会直接生效；若传入自定义配置，则自定义对象作为该次请求的生成配置。

### 10.4 完整生成配置示例

```csharp
var generation = new GenerationConfig
{
    ThinkMode = true,
    ShowThinking = false,
    ThinkingBudget = 192,
    MultiTurn = true,
    MaxTurns = 4,
    MaxHistoryChars = 500,
    MaxHistoryTokens = 1200,
    MaxNewTokens = 384,
    Stream = true,
    Temperature = 0.6,
    TopP = 0.95,
    TopK = 20,
    RepetitionPenalty = 1.05,
    DoSample = true,
    SystemPrompt = "请用准确、简洁的中文回答。",
    StripThinkingInHistory = true,
    SanitizeHistory = true,
    ResetOnFactQuestion = true,
    EnableRepetitionCheck = true,
    RepetitionThreshold = 0.9,
    EnableLoopDetection = true,
    LoopCheckWindow = 20,
    LoopRepeatThreshold = 4
};

var result = assistant.Ask("解释 ONNX Runtime", generationConfig: generation);
```

> `ShowThinking` 目前主要由 CLI 的输出过滤器实现；核心引擎负责生成文本，C# 直接调用者应根据需求决定是否展示 `result.Answer` 中的 think 区段。

### 10.5 流式回调与异常

- `onToken` 用于接收当前请求的逐 token 文本片段。
- 引擎只在 `GenerationConfig.Stream` 为 true 时调用流回调。
- 不应在回调中执行长时间阻塞操作；如需 UI 更新，可转发至应用自己的消息队列/同步上下文。
- 文件缺失、ONNX 图不兼容、配置无效等错误通过异常报告。集成层应按应用日志规范记录并决定是否向用户展示。
- CLI 会对解析和执行异常设置退出码；类库本身不会吞掉推理异常。

### 10.6 多轮、历史与清理

当 `MultiTurn = true` 时，assistant 保存裁剪后的 user/assistant 文本历史，并在请求前重建对应的 KV Cache：

- `MaxTurns` 限制保留轮数。
- `MaxHistoryChars` 限制每轮 assistant 回复的字符长度。
- `MaxHistoryTokens` 超限时丢弃最旧轮次。
- `StripThinkingInHistory` 在保存前提取最终答案部分。
- `SanitizeHistory` 对 roleplay 括号内容做简单清理。
- `ResetOnFactQuestion` 对判定为独立事实问题的提问清空旧历史。
- `ClearMemory()` 清理 assistant 的历史和引擎 cache。

每次改变多轮模式、think 模式或 system prompt 时，当前实现会清空历史以避免不兼容的 KV Cache 继续复用。

### 10.7 结果对象

`Ask` 返回 `AssistantResult`：

| 属性 | 含义 |
| --- | --- |
| `Answer` | 模型文本或重复回答拦截后的提示 |
| `Generation` | 生成 token 数、耗时、速度、循环及 thinking budget 状态 |
| `PromptTokens` | 本次 prompt token 数 |
| `HistoryTurns` | 请求结束后的历史轮数 |
| `FactReset` | 本次是否因事实问题清空过历史 |
| `RepetitionDetected` | 是否检测到重复回答 |
| `ThinkOverride` | 是否通过强制 think 设置覆盖配置 |
| `SceneName` | 场景名称或 `custom` |

`GenerationStats` 的 `TotalTimeSeconds` 计量生成阶段；`TokensPerSecond` 依此计算，不应当作包括模型初始化、tokenizer prefill 和磁盘加载在内的端到端请求时间。

## 11. 扩展到其他模型

### 11.1 扩展边界

模型专有逻辑应放在 `IInferenceEngine` 实现中。接口当前包括：

```csharp
public interface IInferenceEngine : IDisposable
{
    string EngineId { get; }
    long HistoryLength { get; }
    int CountTokens(string text);
    void ResetState();
    void Prefill(string text);
    string FormatSystemPrompt(string systemPrompt);
    string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns);
    string FormatUserPrompt(string question, bool thinkMode);
    (string Text, GenerationStats Stats) Generate(
        string prompt,
        GenerationConfig config,
        Action<string>? onToken = null);
}
```

引擎工厂负责根据 `AssistantConfig.EngineId` 返回实现；`ChatAssistant` 负责编排而不直接创建 ONNX Runtime session。

### 11.2 注册新引擎示例

```csharp
using Qwen3.Inference;

var factory = new InferenceEngineFactory();
factory.Register("another-engine", config => new AnotherModelEngine(config));

var config = new AssistantConfig
{
    EngineId = "another-engine",
    ModelPath = @"D:\models\another-model"
};

using var assistant = new ChatAssistant(config, factory);
var answer = assistant.Ask("你好");
Console.WriteLine(answer.Answer);
```

`AnotherModelEngine` 需要实现 `IInferenceEngine`，实现释放、token 计数、提示词格式化、状态重置、历史 prefill 和生成。它应明确处理取消、资源边界、输出统计及无效模型文件，而不是在适配层静默忽略错误。

如果新模型的对话模板、thinking 语法和历史 cache 语义与 Qwen 不同，应由新 engine 的 `Format*Prompt` 及状态管理负责适配。若仅替换模型文件而网络图结构与当前模型不兼容，也必须采用独立引擎实现。

### 11.3 工厂扩展说明

当前 `InferenceEngineFactory` 在构造时注册内置 `qwen3-onnx`。调用 `Register` 增加其他 ID；重复 ID 会抛出异常。CLI 目前创建默认 factory，因此新增引擎只有在 CLI 注册代码中加入相应 factory 后，`--engine <id>` 才可使用。自定义宿主程序可传入自己配置好的 factory。

## 12. 日志、错误和排障

### 12.1 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 找不到 tokenizer/config/ONNX 文件 | 核对 `--model-path` 指向实际目录，确认文件名和下载完整性 |
| ONNX session 初始化失败 | 检查 ONNX Runtime 原生依赖、图文件、external data 和执行平台 |
| decoder 输入数或名称不匹配 | 该导出图契约不适配当前 `OnnxInferenceEngine`；检查图版本或实现专用引擎 |
| 不支持的张量类型/输出形状 | 当前引擎限制了 logits、KV Cache 和 sequence length 的类型及排列，需要对应扩展实现 |
| 看不到回答只看到日志 | 日志使用 stderr；检查 stdout/stderr 分流以及 `--log-level` |
| 生成提前停止 | 检查 stop token、`--max-new-tokens`、`--thinking-budget` 和 loop detection |
| 输出仍出现推理过程 | `--hide-thinking` 过滤 `<think>...</think>`；模型若未输出这些标签，不能保证识别其他格式 |
| 历史轮次被清除 | 检查场景是否切换了多轮/think/system prompt，或是否启用 fact reset |
| Tokenizer 原生库加载失败 | 确认发布 RID、进程架构与目标机器的 Tokenizers 原生依赖 |

### 12.2 日志安全

模型路径、日志异常堆栈和用户自定义异常信息可能进入日志。请按运行环境策略保护日志文件、检查保留周期，并在提交工单或对外分享前审阅日志。

## 13. 当前限制与后续扩展方向

当前实现的已知边界：

1. ONNX Runtime 使用 CPU Provider。
2. Tokenizers 原生依赖已配置 Windows x64、Linux x64/arm64 和 macOS x64/arm64；仍需在目标系统验证原生库加载。
3. 引擎默认文件名为 `embedding.int8.onnx` 和 `decoder.int8.onnx`；CLI 未提供独立 ONNX 文件名参数。
4. decoder 输入输出契约按当前 Qwen3 导出图布局实现，不保证与所有 ONNX 导出器或量化模型兼容。
5. CLI 是单次请求进程；多轮交互由 `Qwen3.Console` 或类库宿主程序负责。
6. 场景检测和事实问题检测采用规则匹配，不属于语义分类。
7. `ShowThinking` 由 CLI 的文本过滤实现，不控制模型内部生成阶段。

适合后续演进的方向：

- 为 ONNX 图名、外部数据路径、Provider、线程和最大上下文建立明确的模型运行配置。
- 为生成请求增加统一的取消与请求上下文传播；桌面端现已支持在 ONNX token 生成循环中响应停止。
- 为引擎配置/能力提供模型元数据描述，避免在通用对话层假设特定模型的 think 语法。
- 抽出独立的聊天模板/Tokenizer/采样策略接口，以适配不同架构。
- 为 CLI 增加文件名参数、配置文件、批量输入和 JSON 输出模式。
- 建立不依赖大模型权重的单元测试，以及在拥有模型文件时运行的 ONNX 集成测试。

在新增平台、模型格式或 Provider 前，应使用对应运行时和真实模型资产进行完整验证。

## 14. Avalonia 桌面端

桌面端项目为 `Qwen3.Desktop`，使用 Avalonia 12、CommunityToolkit.Mvvm、SQLite 和 `Qwen3.Inference`。界面包含聊天、模型管理、历史记录、用户中心及管理员账户管理；侧栏可折叠。所有模型推理由统一的 `DesktopChatService` 调用核心 `ChatAssistant`，不会另建一套推理实现。

### 14.1 启动与构建

```powershell
dotnet restore Qwen3Onnx.sln
dotnet run --project Qwen3.Desktop
```

发布时指定目标平台 RID，例如：

```powershell
dotnet publish Qwen3.Desktop/Qwen3.Desktop.csproj -c Release -r win-x64 --self-contained false
dotnet publish Qwen3.Desktop/Qwen3.Desktop.csproj -c Release -r linux-x64 --self-contained false
dotnet publish Qwen3.Desktop/Qwen3.Desktop.csproj -c Release -r osx-arm64 --self-contained false
```

跨平台发布需要目标机安装 .NET 8 Runtime；若要生成自包含程序，可将 `--self-contained false` 改为 `--self-contained true`。ONNX Runtime 当前使用 CPU Provider。请在目标系统检查原生依赖、文件权限和窗口行为；单个平台构建成功不代表其他平台的运行环境已验证。

### 14.2 首次启动与数据库

应用在系统用户的本地应用数据目录建立 `Qwen3OnnxDesktop` 文件夹，其中包括 `qwen3-desktop.sqlite3` 数据库、`models` 模型目录和 `logs\desktop.log` 日志文件。具体根目录遵循 .NET `Environment.SpecialFolder.LocalApplicationData`；Windows 通常位于 `%LOCALAPPDATA%\Qwen3OnnxDesktop`。

首次初始化数据库和表结构，并预置用户提供的四个 ModelScope 模型条目。若尚无管理员账户，程序创建用户名为 `admin` 的账户，使用密码学安全随机数生成一次性密码，并在启动界面显示。请先妥善保存该密码再关闭提示；密码仅以 PBKDF2-SHA256 哈希和 salt 存储，不会在后续启动时再次显示。当前没有管理员密码重置流程。

### 14.3 Chat 与文本附件

1. 在模型管理中下载至少一个模型，然后回到聊天顶部选择已安装模型。
2. 输入文本，选择“思考模式”并按需打开“高级配置”调整生成长度、温度、Top P/Top K、重复惩罚、思考预算、历史轮数、采样和流式输出。
3. 附件只作为文本输入拼入提示词，不会作为二进制文件发送。支持 `.txt`、`.md`、`.json`、`.csv`、`.xml`、常见配置文件及部分源码格式；每条消息最多 5 个文件，每个文件最大 1 MiB。非文本内容和不支持的扩展名会拒绝。
4. 首轮回答后，桌面端尝试用模型生成简短标题；若标题生成失败，会回退为首条输入的摘要。消息、附件名/内容、生成 token 数和标题会存入 SQLite。
5. 消息正文按 Markdown 渲染，支持常用标题、列表、引用、链接、行内格式和代码块。消息下方的图标工具栏提供复制、重新发送或重新生成回复；悬停图标可查看提示。消息文本可直接选择复制；助手消息可切换到可选择文本模式（此模式显示 Markdown 源文本）。复制按钮复制正文文本，不包含单独呈现的思考段。
6. 助手回复中的 `<think>...</think>` 单独显示在可折叠区域，收起时以约 6 行高度跟随显示最新输出，生成期间显示思考状态；展开后可滚动查看思考文本。最新一条助手回复可点“重新生成回复”；程序使用该轮之前的历史重新推理，成功后替换原回复，失败则保留原回复。重新生成产生的 token 用量会继续计入统计。
7. 生成期间“发送”按钮变为“停止”。达到思考预算后会结束思考段，并额外保留完整的“输出长度”配额继续生成回答。停止会尽快终止 token 循环，并将当前用户输入、部分回答及中断标记一并保存到 SQLite；侧栏标题、历史重开和后续上下文均保留本轮记录。停止重新生成时，已生成的部分内容会替换原回复并带有中断标记。

数据库消息和 token 用量按一轮请求原子写入；推理失败时不会留下空会话或半条消息，并会保留输入供重试。重新打开对话时会从数据库消息重建推理历史。旧数据库会自动增加消息中断标记字段。

状态栏的 `tokens/s` 是 ONNX 引擎实际生成阶段的速度，与 CLI 的推理统计使用相同计时口径；`本轮` 则额外包括模型首次加载、历史 KV cache 重建、首轮标题生成和数据库保存，因此不应直接拿本轮秒数与 CLI 生成耗时对比。桌面端默认采用聊天场景参数（思考预算 192、答案最多生成 256 tokens、最多保留 3 轮历史）；思考预算是额外的思考上限，达到后结束思考并继续使用完整答案长度额度。CLI 会根据问题自动选择场景，例如事实类问题可能使用不思考的快速问答配置，所以只有场景、思考模式和生成参数一致时，`tokens/s` 才适合横向比较。生成过程中使用轻量纯文本控件并每 100ms 批量刷新，避免持续解析和重绘 Markdown；完成后再切换为 Markdown 渲染。

桌面聊天默认保留最近最多 3 轮对话作为上下文，不会因为识别为事实类问题而自动清空历史。状态栏显示的“上下文 N 轮”是本次回答使用的先前对话轮数；重新打开历史会话时会从 SQLite 恢复完整消息，再按轮数和 token 限额裁剪后重建推理上下文。

### 14.4 模型登记、下载和删除

模型管理显示内置模型和管理员添加的模型。管理员登记或编辑模型标题、英文目录名、描述、Git clone 仓库 URL，以及 HTTP 备用文件清单。HTTP 清单每行填写相对路径、该文件 URL，并可选填写预期大小和 SHA-256，例如：

```text
config.json | https://host.example/model/config.json | 420 | <sha256>
tokenizer.json | https://host.example/model/tokenizer.json
embedding.int8.onnx | https://host.example/model/embedding.int8.onnx
decoder.int8.onnx | https://host.example/model/decoder.int8.onnx
```

相对路径保持模型仓库内的目录结构；必需模型文件应包含 `config.json`、`tokenizer.json`、`embedding*.onnx` 和 `decoder*.onnx`。下载优先尝试 Git 浅克隆；Git 不存在、克隆失败、关键文件缺失或 Git LFS 权重仅得到指针时，会保留已克隆的完整文件，并使用登记的逐文件 HTTP 链接补齐。如果未登记这些链接，内置 ModelScope 模型会继续使用 ModelScope 文件清单 API 下载。HTTP 下载会校验安全相对路径、可用的文件大小和 SHA-256；必需文件校验通过后才发布模型目录。发布目录后会尽力清理 `.git` 元数据；若操作系统拒绝删除 Git pack 文件，模型仍会安装成功，`.git` 会留在模型目录中并在状态中提示。下载或校验失败时，临时 `.partial` 目录会保留并在状态提示中显示其路径，便于排查；再次开始下载会使用新的临时目录，不会自动续传旧目录。

模型列表提供编辑与删除。添加或编辑会在遮罩弹窗中完成，删除前会显示确认弹窗。已安装模型或已被对话引用的模型不能更改英文名称（避免本地路径/历史关联失效）；被历史对话引用的模型不能删除。

HTTP 后备目前仅实现 ModelScope 文件清单协议。其他 Git 兼容站点在系统安装 Git 时可尝试克隆；如果 Git 不可用，目前不能自动解析任意站点的 HTTP 文件清单。添加未知模型前，应确认仓库结构、ONNX 图输入输出及 tokenizer/config 契约符合当前推理核心。删除仍被历史会话引用的模型会被拒绝，以避免破坏历史对话关联。

### 14.5 历史、登录与权限范围

- 访客对话使用 `owner_user_id IS NULL`，对本机访客可见。
- 侧栏始终显示当前访问身份最近更新的最多 100 条对话标题，点击即可继续聊天。
- 登录用户可在左侧导航栏底部的“账户与更多”菜单打开“管理我的对话”页面，检索、修改标题、删除和继续自己的全部会话。该页面不会跨账户读取对话内容；管理员也只能管理自己的会话。
- 用户切换身份时聊天和侧栏历史会清空并重新加载该身份的数据。生成进行中会禁用新对话和身份切换，避免跨身份混合。
- 管理员可创建本地用户、查看累计 token 用量和锁定/解锁普通账户。管理员不会获得读取用户私聊内容的查询接口。
- 密码至少 12 个字符，以 PBKDF2-SHA256 派生值存储。锁定账户不能重新登录；本地数据库和模型文件仍由当前操作系统账户管理，建议保护系统账户和应用数据目录。
- “账户与更多”菜单同时提供用户中心、账户切换/退出、项目官网和关于页面。官网使用系统默认浏览器打开 `https://manyeyes.github.io/manyspeech/`；“关于”页面展示产品介绍、应用版本和 .NET Runtime 信息。

桌面端账户和会话是单机本地功能，不是远程身份认证或多设备同步服务。管理员对本地 SQLite 文件和程序进程的操作权限属于本机安全边界。

### 14.6 自助注册与用户中心

访客点击侧栏“登录 / 切换账户”，再选择“没有账户？立即注册”，填写用户名和两次密码即可创建普通账户并自动登录。用户名为 3–64 个字母、数字、点、下划线或连字符；密码至少 12 个字符。注册不会开放管理员角色。

登录后从侧栏进入“用户中心”，可维护昵称、头像 URL、生日（`yyyy-MM-dd`）、博客链接和个人简介，并可凭当前密码修改密码。头像 URL 只保存为个人资料，不会作为模型附件或上传到远端服务；用户头像区域使用本地首字母占位展示。

“我的画像”仅查询当前登录用户的会话、消息和 token 用量。关键词通过本地文本统计提取，领域分布依据少量内置词表估算，仅用于概览，不代表模型生成的语义画像，也不会读取其他用户的私聊。删除账户需要输入当前密码；管理员账户不可自助注销，以免移除本机管理入口。注销会删除该用户、个人资料和其拥有的私有对话/消息；历史用量事件按数据库外键解除账户关联，不再计入任何用户的画像。访客对话和模型文件不受影响。
