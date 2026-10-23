using Qwen3.Inference;

var modelPath = args.Length > 0
    ? Path.GetFullPath(args[0])
    : Environment.GetEnvironmentVariable("QWEN_MODEL_PATH") ??
      Path.Combine(AppContext.BaseDirectory, "models", "Qwen3-0.6b-int8-onnx");

var assistantConfig = new AssistantConfig
{
    EngineId = "qwen3-onnx",
    ModelPath = modelPath,
    DecoderModelName = ResolveModelFile(modelPath, "decoder"),
    EmbeddingModelName = ResolveModelFile(modelPath, "embedding"),
    SystemPrompt = "You are a helpful assistant."
};
var requiredModelFiles = new[]
{
    assistantConfig.TokenizerPath,
    assistantConfig.ModelConfigPath,
    assistantConfig.EmbeddingModelPath,
    assistantConfig.DecoderModelPath
};
var missingModelFiles = requiredModelFiles.Where(path => !File.Exists(path)).ToArray();
if (missingModelFiles.Length > 0)
{
    Console.Error.WriteLine("模型目录缺少以下文件：");
    foreach (var path in missingModelFiles)
    {
        Console.Error.WriteLine($"  {path}");
    }
    Environment.ExitCode = 1;
    return;
}

try
{
    using var assistant = new ChatAssistant(assistantConfig);
    RunChat(assistant, assistantConfig);
}
catch (Exception exception)
{
    Console.Error.WriteLine($"模型初始化或推理失败：{exception.Message}");
    Environment.ExitCode = 1;
}

static void RunChat(ChatAssistant assistant, AssistantConfig config)
{
    bool? globalThinkOverride = null;
    PrintHelp();
    while (true)
    {
        var status = globalThinkOverride switch
        {
            true => "强制开启",
            false => "强制关闭",
            _ => "自动"
        };
        Console.Write($"\n你[{status}]: ");
        var input = Console.ReadLine();
        if (input is null)
        {
            break;
        }

        var question = input.Trim();
        if (question.Length == 0)
        {
            continue;
        }

        switch (question.ToLowerInvariant())
        {
            case "exit":
            case "quit":
            case "q":
                Console.WriteLine("再见！");
                return;
            case "clear":
                Console.Clear();
                continue;
            case "reset":
                assistant.ClearMemory();
                Console.WriteLine("[已清空对话历史 + KV Cache]");
                continue;
            case "memory":
                PrintMemory(assistant);
                continue;
            case "scenes":
                PrintScenes();
                continue;
            case "config":
                Console.WriteLine($"模型路径: {config.ModelPath}");
                Console.WriteLine($"最大序列长度: {config.MaxSequenceLength}");
                Console.WriteLine($"线程数: intra_op={config.IntraOpNumThreads}, inter_op={config.InterOpNumThreads}");
                Console.WriteLine($"System prompt: {config.SystemPrompt}");
                Console.WriteLine($"全局思考模式: {status}");
                continue;
            case "/think on":
            case "/think true":
            case "/think 1":
                globalThinkOverride = true;
                assistant.ClearMemory();
                Console.WriteLine("[已全局强制开启思考模式，历史已清空]");
                continue;
            case "/think off":
            case "/think false":
            case "/think 0":
                globalThinkOverride = false;
                assistant.ClearMemory();
                Console.WriteLine("[已全局强制关闭思考模式，历史已清空]");
                continue;
            case "/think":
            case "/think auto":
            case "/think default":
                globalThinkOverride = null;
                assistant.ClearMemory();
                Console.WriteLine("[思考模式已恢复自动检测，历史已清空]");
                continue;
        }

        try
        {
            var thinkOverride = globalThinkOverride;
            if (question.StartsWith("think:", StringComparison.OrdinalIgnoreCase))
            {
                var parts = question.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 2 || !TryParseThinkMode(parts[0][6..], out thinkOverride))
                {
                    Console.WriteLine("[用法: think:on <问题>、think:off <问题> 或 think:auto <问题>]");
                    continue;
                }
                question = parts[1];
            }

            Scene? sceneOverride = null;
            if (question.StartsWith("scene:", StringComparison.OrdinalIgnoreCase))
            {
                var parts = question.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 2)
                {
                    Console.WriteLine("[用法: scene:fast_qa <问题>]");
                    continue;
                }
                if (!GenerationConfig.TryParseScene(parts[0][6..], out var selectedScene))
                {
                    Console.WriteLine("[可用场景: fast_qa, deep, roleplay, precise, chitchat]");
                    continue;
                }
                sceneOverride = selectedScene;
                question = parts[1];
                Console.WriteLine($"[使用场景: {GenerationConfig.GetSceneName(selectedScene)}]");
            }

            Console.Write("助手: ");
            using var tokenWriter = new BufferedConsoleWriter(Console.Out);
            var result = assistant.Ask(question, sceneOverride, thinkOverride,
                token => tokenWriter.Write(token));
            tokenWriter.Flush();
            Console.WriteLine();
            PrintStats(result);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"\n本次推理失败：{exception.Message}");
        }
    }
}

static bool TryParseThinkMode(string value, out bool? mode)
{
    switch (value.ToLowerInvariant())
    {
        case "on":
        case "true":
        case "1":
            mode = true;
            return true;
        case "off":
        case "false":
        case "0":
            mode = false;
            return true;
        case "auto":
        case "default":
            mode = null;
            return true;
        default:
            mode = null;
            return false;
    }
}

static void PrintStats(AssistantResult result)
{
    var stats = result.Generation;
    Console.WriteLine(new string('-', 60));
    Console.WriteLine($"生成 token 数: {stats.TotalTokens} | 总耗时: {stats.TotalTimeSeconds:F2}s | 速度: {stats.TokensPerSecond:F1} tokens/s");
    Console.WriteLine($"prompt 长度: {result.PromptTokens} tokens | 历史轮数: {result.HistoryTurns}");
    if (stats.LoopDetected)
    {
        Console.WriteLine("[触发: 复读循环]");
    }
    if (result.FactReset)
    {
        Console.WriteLine("[触发: 事实问题清空历史]");
    }
    if (result.RepetitionDetected)
    {
        Console.WriteLine("[检测到整段复读，已丢弃本次回答]");
    }
    if (stats.ThinkingBudgetExceeded)
    {
        Console.WriteLine("[触发: thinking_budget 超限，强制结束思考]");
    }
    if (result.ThinkOverride.HasValue)
    {
        Console.WriteLine($"[用户指定: think_mode={result.ThinkOverride.Value}]");
    }
    Console.WriteLine(new string('-', 60));
}

static void PrintMemory(ChatAssistant assistant)
{
    Console.WriteLine($"[当前历史轮数: {assistant.History.Count}]");
    for (var index = 0; index < assistant.History.Count; index++)
    {
        var turn = assistant.History[index];
        Console.WriteLine($"  {index + 1}. 你: {turn.User}");
        Console.WriteLine($"     助手: {turn.Assistant[..Math.Min(80, turn.Assistant.Length)]}");
    }
}

static void PrintScenes()
{
    Console.WriteLine("[可用场景]");
    foreach (var (scene, config) in GenerationConfig.SceneConfigs)
    {
        Console.WriteLine($"  {GenerationConfig.GetSceneName(scene)}: think_mode={config.ThinkMode}, " +
                          $"max_new_tokens={config.MaxNewTokens}, multi_turn={config.MultiTurn}, " +
                          $"temperature={config.Temperature}, top_p={config.TopP}, top_k={config.TopK}");
    }
}

static void PrintHelp()
{
    Console.WriteLine("Qwen3 问答助手");
    Console.WriteLine("命令: exit/quit/q 退出 | reset 清空记忆 | memory 查看历史 | scenes 查看场景 | config 查看配置");
    Console.WriteLine("思考模式: think:on <问题> | think:off <问题> | /think on | /think off | /think auto");
    Console.WriteLine("指定场景: scene:fast_qa <问题>（也支持 deep、roleplay、precise、chitchat）");
    Console.WriteLine("直接输入问题将自动判断场景。");
}

static string ResolveModelFile(string modelPath, string component)
{
    var preferW4A32 = modelPath.Contains("w4a32", StringComparison.OrdinalIgnoreCase);
    var preferInt8 = modelPath.Contains("int8", StringComparison.OrdinalIgnoreCase);
    var variants = preferW4A32
        ? new[] { "w4a32", "int8", "int4" }
        : preferInt8
            ? new[] { "int8", "w4a32", "int4" }
            : new[] { "int4", "w4a32", "int8" };
    foreach (var variant in variants)
    {
        var fileName = $"{component}.{variant}.onnx";
        if (File.Exists(Path.Combine(modelPath, fileName)))
        {
            return fileName;
        }
    }

    throw new FileNotFoundException(
        $"模型目录缺少受支持的 {component} ONNX 文件。",
        Path.Combine(modelPath, $"{component}.{variants[0]}.onnx"));
}

sealed class BufferedConsoleWriter : TextWriter
{
    private readonly TextWriter _writer;
    private readonly System.Text.StringBuilder _buffer = new();
    private readonly object _gate = new();
    private readonly Timer _timer;
    private bool _disposed;

    public BufferedConsoleWriter(TextWriter writer)
    {
        _writer = writer;
        _timer = new Timer(_ => Flush(), null, 50, 50);
    }

    public override System.Text.Encoding Encoding => _writer.Encoding;

    public override void Write(string? value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return;
        }
        lock (_gate)
        {
            if (!_disposed)
            {
                _buffer.Append(value);
            }
        }
    }

    public override void Flush()
    {
        lock (_gate)
        {
            if (_buffer.Length == 0)
            {
                return;
            }
            _writer.Write(_buffer.ToString());
            _buffer.Clear();
            _writer.Flush();
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            lock (_gate)
            {
                if (!_disposed)
                {
                    _disposed = true;
                    _timer.Dispose();
                    FlushBuffer();
                }
            }
        }
        base.Dispose(disposing);
    }

    private void FlushBuffer()
    {
        if (_buffer.Length == 0)
        {
            return;
        }
        _writer.Write(_buffer.ToString());
        _buffer.Clear();
        _writer.Flush();
    }
}
