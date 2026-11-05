namespace Qwen3.Inference;

public sealed class AssistantConfig
{
    /// <summary>引擎标识：qwen3-onnx（旧）或 qwen3-onnx-genai（新）。</summary>
    public string EngineId { get; init; } = "qwen3-onnx-genai";

    /// <summary>
    /// 模型目录。
    /// - GenAI 引擎：指向包含 genai_config.json 的目录。
    /// - 旧 ONNX 引擎：指向包含 embedding/decoder 的目录。
    /// </summary>
    public string ModelPath { get; init; } =
        Path.Combine(AppContext.BaseDirectory, "models", "qwen3-4b-genai-int4");

    // ==================== 旧 ONNX 引擎专用（GenAI 引擎忽略） ====================
    public string EmbeddingModelName { get; init; } = "embedding.int8.onnx";
    public string DecoderModelName { get; init; } = "decoder.int8.onnx";
    public int IntraOpNumThreads { get; init; } = Math.Max(1, Environment.ProcessorCount / 2);
    public int InterOpNumThreads { get; init; } = 1;

    // ==================== GenAI 引擎专用 ====================
    /// <summary>GenAI 模型目录，默认与 ModelPath 相同。</summary>
    public string GenaiModelPath { get; init; } = string.Empty;

    /// <summary>是否启用 CUDA provider。false 时使用 CPU。</summary>
    public bool UseCuda { get; init; } = false;

    /// <summary>系统提示词，由引擎在拼接 prompt 时使用。</summary>
    public string SystemPrompt { get; init; } = "You are a helpful assistant.";

    /// <summary>KV Cache 上限，GenAI 内部用于限制生成长度。</summary>
    public int MaxSequenceLength { get; init; } = 4096;

    // ==================== 派生路径 ====================

    /// <summary>GenAI 实际加载目录：优先 GenaiModelPath，否则回退 ModelPath。</summary>
    public string ResolvedGenaiModelPath =>
        string.IsNullOrEmpty(GenaiModelPath) ? ModelPath : GenaiModelPath;

    /// <summary>旧 ONNX 引擎的 tokenizer.json 路径（GenAI 用 model 自带的 tokenizer）。</summary>
    public string TokenizerPath => Path.Combine(ModelPath, "tokenizer.json");

    public string EmbeddingModelPath => Path.Combine(ModelPath, EmbeddingModelName);
    public string DecoderModelPath => Path.Combine(ModelPath, DecoderModelName);
    public string ModelConfigPath => Path.Combine(ModelPath, "config.json");

    /// <summary>GenAI 的 genai_config.json 路径，用于启动时快速校验。</summary>
    public string GenaiConfigPath => Path.Combine(ResolvedGenaiModelPath, "genai_config.json");
}

public enum Scene
{
    FastQa,
    DeepReasoning,
    Roleplay,
    Precise,
    Chitchat
}

public sealed record GenerationConfig
{
    public bool ThinkMode { get; init; } = true;
    public bool ShowThinking { get; init; } = true;
    public int ThinkingBudget { get; init; } = 256;
    public bool MultiTurn { get; init; } = true;
    public int MaxTurns { get; init; } = 3;
    public int MaxHistoryChars { get; init; } = 400;
    public int MaxHistoryTokens { get; init; } = 800;
    public int MaxNewTokens { get; init; } = 256;
    public bool Stream { get; init; } = true;
    public double Temperature { get; init; } = 0.6;
    public double TopP { get; init; } = 0.95;
    public int TopK { get; init; } = 20;
    public double RepetitionPenalty { get; init; } = 1.1;
    public bool DoSample { get; init; } = true;
    public string? SystemPrompt { get; init; }
    public bool StripThinkingInHistory { get; init; } = true;
    public bool SanitizeHistory { get; init; } = true;
    public bool ResetOnFactQuestion { get; init; } = true;
    public bool EnableRepetitionCheck { get; init; } = true;
    public double RepetitionThreshold { get; init; } = 0.9;
    public bool EnableLoopDetection { get; init; } = true;
    public int LoopCheckWindow { get; init; } = 20;
    public int LoopRepeatThreshold { get; init; } = 4;

    public static IReadOnlyDictionary<Scene, GenerationConfig> SceneConfigs { get; } =
        new Dictionary<Scene, GenerationConfig>
        {
            [Scene.FastQa] = new()
            {
                ThinkMode = false,
                ThinkingBudget = 0,
                MultiTurn = false,
                MaxNewTokens = 192,
                Temperature = 0.2,
                TopP = 0.9,
                TopK = 20,
                RepetitionPenalty = 1.05,
                ResetOnFactQuestion = false
            },
            [Scene.DeepReasoning] = new()
            {
                ThinkMode = true,
                ThinkingBudget = 384,
                MultiTurn = false,
                MaxNewTokens = 768,
                Temperature = 0.4,
                TopP = 0.95,
                TopK = 20,
                RepetitionPenalty = 1.05,
                ResetOnFactQuestion = false
            },
            [Scene.Roleplay] = new()
            {
                ThinkMode = true,
                ThinkingBudget = 256,
                MultiTurn = true,
                MaxTurns = 5,
                MaxHistoryChars = 600,
                MaxHistoryTokens = 1536,
                MaxNewTokens = 384,
                Temperature = 0.7,
                TopP = 0.95,
                TopK = 40,
                RepetitionPenalty = 1.1,
                ResetOnFactQuestion = false
            },
            [Scene.Precise] = new()
            {
                ThinkMode = false,
                ThinkingBudget = 0,
                MultiTurn = false,
                MaxNewTokens = 512,
                Temperature = 0.15,
                TopP = 0.9,
                TopK = 10,
                RepetitionPenalty = 1.0,
                DoSample = false,
                ResetOnFactQuestion = false
            },
            [Scene.Chitchat] = new()
            {
                ThinkMode = true,
                ThinkingBudget = 192,
                MultiTurn = true,
                MaxTurns = 3,
                MaxHistoryChars = 400,
                MaxHistoryTokens = 1024,
                MaxNewTokens = 256,
                Temperature = 0.6,
                TopP = 0.95,
                TopK = 20,
                RepetitionPenalty = 1.05
            }
        };

    public static string GetSceneName(Scene scene) => scene switch
    {
        Scene.FastQa => "fast_qa",
        Scene.DeepReasoning => "deep",
        Scene.Roleplay => "roleplay",
        Scene.Precise => "precise",
        Scene.Chitchat => "chitchat",
        _ => throw new ArgumentOutOfRangeException(nameof(scene))
    };

    public static Scene DetectScene(string question) => TextUtilities.DetectScene(question);

    public static bool TryParseScene(string value, out Scene scene)
    {
        scene = value.ToLowerInvariant() switch
        {
            "fast_qa" => Scene.FastQa,
            "deep" => Scene.DeepReasoning,
            "roleplay" => Scene.Roleplay,
            "precise" => Scene.Precise,
            "chitchat" => Scene.Chitchat,
            _ => (Scene)(-1)
        };
        return Enum.IsDefined(scene);
    }
}
