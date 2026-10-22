# Qwen3 ONNX Inference for .NET

使用 C#、ONNX Runtime 在本地运行 Qwen3 INT8 / W4A32 ONNX 模型。方案包含可复用的推理核心、交互式控制台和独立 CLI，支持场景化生成参数、KV Cache、对话历史、日志及可扩展的推理引擎接口。

> CLI 和交互式控制台直接使用本地模型文件；桌面端可在模型管理中按需下载内置 ModelScope 模型。模型权重不包含在代码仓库中。

## 功能

- **本地 ONNX 推理**：拆分 embedding 与 decoder，使用 ONNX Runtime CPU Execution Provider。
- **模型推理核心**：统一的 `ChatAssistant` API、`IInferenceEngine` 接口和可注册的 `IInferenceEngineFactory`。
- **两种应用入口**：
  - `Qwen3.Cli`：单次问答命令行，提供参数解析、校验、帮助、日志和退出码。
  - `Qwen3.Console`：交互式问答，提供多轮记忆、场景选择和思考模式控制。
- **Avalonia 桌面应用**：`Qwen3.Desktop` 使用 MVVM 页面切换，提供本地聊天、文本附件、模型 Git/HTTP 下载、SQLite 历史、本地账户注册与用户中心；支持停止生成，侧栏展示最近 100 条记录，登录用户可管理自己的对话和个人资料。
- **生成控制**：温度、Top-P / Top-K、重复惩罚、思考预算、输出长度、流式输出及复读/循环检测。
- **场景配置**：`fast_qa`、`deep`、`roleplay`、`precise`、`chitchat`。
- **本地日志**：CLI 标准错误日志及可选 JSON Lines 日志文件。

## Qwen3 ONNX 模型

本项目面向 Qwen3 文本生成模型的本地 ONNX 部署。Qwen3 模型支持思考/非思考式对话使用方式；本项目通过 Qwen ChatML 提示词和 `think` 标签支持对应控制。是否实际生成思考文本取决于所用 checkpoint、提示词和生成参数。

| 参数规模 | INT8 | W4A32 |
| --- | --- | --- |
| 0.6B | [ModelScope](https://modelscope.cn/models/manyeyes/Qwen3-0.6B-int8-onnx) | [ModelScope](https://modelscope.cn/models/manyeyes/Qwen3-0.6B-w4a32-onnx) |
| 1.7B | [ModelScope](https://modelscope.cn/models/manyeyes/Qwen3-1.7B-int8-onnx) | [ModelScope](https://modelscope.cn/models/manyeyes/Qwen3-1.7B-w4a32-onnx) |

请在 ModelScope 模型页面下载模型文件。下载后的文件组织方式可能随模型仓库版本变化；请确认 tokenizer、`config.json`、embedding ONNX、decoder ONNX 和所需的外部 ONNX 数据文件完整。

当前引擎默认读取：

```text
tokenizer.json
config.json
embedding.int8.onnx
decoder.int8.onnx
```

`AssistantConfig.EmbeddingModelName` 和 `DecoderModelName` 可通过 C# 集成代码调整。CLI 目前使用上述默认文件名，没有提供模型文件名单独覆盖选项。W4A32 模型必须同时满足当前引擎的 ONNX 输入/输出与 KV Cache 契约；仅将 W4A32 权重改名为 INT8 文件名并不能保证兼容。若图结构或量化节点不兼容，需要扩展或实现另一种 `IInferenceEngine`。

## 项目结构

```text
Qwen3Onnx.sln
├── Qwen3.Inference/   推理抽象、Qwen3 ONNX 引擎、采样和对话管理
├── Qwen3.Cli/         System.CommandLine 单次问答 CLI
├── Qwen3.Console/     交互式控制台
└── Qwen3.Desktop/     Avalonia + MVVM 跨平台桌面应用
```

## 快速开始

### 环境

- .NET 8 SDK
- Windows、Linux 或 macOS（桌面发布请使用对应的 .NET 8 Runtime）
- 与当前推理引擎输入输出契约兼容的模型文件

### 构建

```powershell
dotnet restore Qwen3Onnx.sln
dotnet build Qwen3Onnx.sln
```

### 桌面端

```powershell
dotnet run --project Qwen3.Desktop
```

首次启动会生成一次性管理员密码并在应用中显示；请立即保存。桌面端功能、模型安装方式、SQLite 数据位置、账户隔离和发布说明见[桌面端使用指南](docs/USER_GUIDE.md#14-avalonia-桌面端)。

### CLI

```powershell
dotnet run --project Qwen3.Cli -- --help

dotnet run --project Qwen3.Cli -- "解释 KV Cache 的作用" `
  --model-path "D:\models\Qwen3-0.6B-int8-onnx" `
  --scene deep `
  --max-new-tokens 512 `
  --temperature 0.4 `
  --top-p 0.95 `
  --top-k 20 `
  --thinking-budget 256 `
  --log-file ".\logs\qwen.jsonl"
```

也可以通过环境变量指定默认模型目录：

```powershell
$env:QWEN_MODEL_PATH = "D:\models\Qwen3-0.6B-int8-onnx"
dotnet run --project Qwen3.Cli -- "你好" --scene chitchat
```

### 交互式控制台

```powershell
dotnet run --project Qwen3.Console -- "D:\models\Qwen3-0.6B-int8-onnx"
```

`Qwen3.Console` 的第一个位置参数是模型目录；也可使用 `QWEN_MODEL_PATH`。启动后输入问题开始对话，输入 `q`、`quit` 或 `exit` 退出。

## CLI 参数速览

| 类别 | 参数 |
| --- | --- |
| 输入与模型 | `<question>`、`--model-path`、`--engine`、`--scene`、`--system-prompt` |
| 思考模式 | `--think-mode auto\|on\|off` |
| 采样 | `--max-new-tokens`、`--temperature`、`--top-p`、`--top-k`、`--sample` / `--no-sample`、`--repetition-penalty` |
| 思考与历史 | `--thinking-budget`、`--show-thinking` / `--hide-thinking`、`--multi-turn` / `--single-turn`、`--max-turns`、`--max-history-chars`、`--max-history-tokens` |
| 质量保护 | `--repetition-check` / `--no-repetition-check`、`--repetition-threshold`、`--loop-detection` / `--no-loop-detection`、`--loop-check-window`、`--loop-repeat-threshold` |
| 历史策略 | `--strip-thinking-in-history` / `--keep-thinking-in-history`、`--sanitize-history` / `--no-sanitize-history`、`--reset-on-fact-question` / `--no-reset-on-fact-question` |
| 输出与日志 | `--stream` / `--no-stream`、`--log-level`、`--log-file` |
| 帮助 | `--help`、`-h`、`-?`、`--version` |

完整参数、默认值、取值范围、模型准备步骤和 C# API 示例请阅读[详细使用与集成指南](docs/USER_GUIDE.md)。

## 推理 API 示例

```csharp
using Qwen3.Inference;

var assistantConfig = new AssistantConfig
{
    EngineId = "qwen3-onnx",
    ModelPath = @"D:\models\Qwen3-0.6B-int8-onnx",
    SystemPrompt = "You are a helpful assistant."
};

var generationConfig = GenerationConfig.SceneConfigs[Scene.DeepReasoning] with
{
    MaxNewTokens = 512,
    Temperature = 0.4,
    TopP = 0.95,
    TopK = 20,
    ThinkingBudget = 256,
    Stream = true
};

using var assistant = new ChatAssistant(assistantConfig);
var result = assistant.Ask(
    "解释 KV Cache 的作用",
    scene: Scene.DeepReasoning,
    generationConfig: generationConfig,
    onToken: Console.Write);

Console.WriteLine();
Console.WriteLine($"生成 token：{result.Generation.TotalTokens}");
Console.WriteLine($"生成速度：{result.Generation.TokensPerSecond:F1} tokens/s");
```

## 扩展模型

统一入口通过 `IInferenceEngine` 封装模型相关操作，包括 tokenizer 计数、提示词格式、cache 状态和生成。通过 `IInferenceEngineFactory` 注册模型实现后，调用端仍可使用 `ChatAssistant`。设计说明和完整扩展示例见[详细指南](docs/USER_GUIDE.md#11-扩展到其他模型)。

## 配置与依赖

- 目标框架：.NET 8
- 命令行：`System.CommandLine 2.0.13`
- 推理：`Microsoft.ML.OnnxRuntime 1.23.2`
- Tokenizer：`Tokenizers.DotNet 1.4.1`
- 日志：`Microsoft.Extensions.Logging.Console 9.0.0`

当前 ONNX 推理实现显式使用 CPU Execution Provider；推理工程配置了 Windows x64、Linux x64/arm64 和 macOS x64/arm64 的 Tokenizers 原生运行时。跨平台发布与目标机运行仍需按相应 RID 验证。

## 文档

- [详细使用与集成指南](docs/USER_GUIDE.md)
- [ModelScope：Qwen3-0.6B INT8 ONNX](https://modelscope.cn/models/manyeyes/Qwen3-0.6B-int8-onnx)
- [ModelScope：Qwen3-0.6B W4A32 ONNX](https://modelscope.cn/models/manyeyes/Qwen3-0.6B-w4a32-onnx)
- [ModelScope：Qwen3-1.7B INT8 ONNX](https://modelscope.cn/models/manyeyes/Qwen3-1.7B-int8-onnx)
- [ModelScope：Qwen3-1.7B W4A32 ONNX](https://modelscope.cn/models/manyeyes/Qwen3-1.7B-w4a32-onnx)

## 免责声明

本项目提供本地模型推理代码，不包含模型权重。请在下载、使用或再分发模型前，阅读并遵守对应模型仓库和上游模型的许可证及使用条款。模型输出由模型生成，使用方应自行评估其准确性和适用性。
