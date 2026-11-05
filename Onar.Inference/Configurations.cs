namespace Onar.Inference;

public sealed class AssistantConfig
{
    /// <summary>�����ʶ��qwen3-onnx���ɣ��� qwen3-onnx-genai���£���</summary>
    public string EngineId { get; init; } = "qwen3-onnx-genai";

    /// <summary>
    /// ģ��Ŀ¼��
    /// - GenAI ���棺ָ����� genai_config.json ��Ŀ¼��
    /// - �� ONNX ���棺ָ����� embedding/decoder ��Ŀ¼��
    /// </summary>
    public string ModelPath { get; init; } =
        Path.Combine(AppContext.BaseDirectory, "models", "qwen3-4b-genai-int4");

    // ==================== �� ONNX ����ר�ã�GenAI ������ԣ� ====================
    public string EmbeddingModelName { get; init; } = "embedding.int8.onnx";
    public string DecoderModelName { get; init; } = "decoder.int8.onnx";
    public int IntraOpNumThreads { get; init; } = Math.Max(1, Environment.ProcessorCount / 2);
    public int InterOpNumThreads { get; init; } = 1;

    // ==================== GenAI ����ר�� ====================
    /// <summary>GenAI ģ��Ŀ¼��Ĭ���� ModelPath ��ͬ��</summary>
    public string GenaiModelPath { get; init; } = string.Empty;

    /// <summary>�Ƿ����� CUDA provider��false ʱʹ�� CPU��</summary>
    public bool UseCuda { get; init; } = false;

    /// <summary>ϵͳ��ʾ�ʣ���������ƴ�� prompt ʱʹ�á�</summary>
    public string SystemPrompt { get; init; } = "You are a helpful assistant.";

    /// <summary>KV Cache ���ޣ�GenAI �ڲ������������ɳ��ȡ�</summary>
    public int MaxSequenceLength { get; init; } = 4096;

    // ==================== ����·�� ====================

    /// <summary>GenAI ʵ�ʼ���Ŀ¼������ GenaiModelPath��������� ModelPath��</summary>
    public string ResolvedGenaiModelPath =>
        string.IsNullOrEmpty(GenaiModelPath) ? ModelPath : GenaiModelPath;

    /// <summary>�� ONNX ����� tokenizer.json ·����GenAI �� model �Դ��� tokenizer����</summary>
    public string TokenizerPath => Path.Combine(ModelPath, "tokenizer.json");

    public string EmbeddingModelPath => Path.Combine(ModelPath, EmbeddingModelName);
    public string DecoderModelPath => Path.Combine(ModelPath, DecoderModelName);
    public string ModelConfigPath => Path.Combine(ModelPath, "config.json");

    /// <summary>GenAI �� genai_config.json ·�����������ʱ����У�顣</summary>
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
