using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Onar.Inference;

public abstract class OnnxInferenceChatTemplateGenaiEngine : IInferenceEngine, IConversationPromptFormatter
{
    private readonly AssistantConfig _config;
    private readonly ILogger? _logger;
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly string _chatTemplate;
    private readonly HashSet<int> _stopTokenIds = [];
    private readonly System.Text.StringBuilder _pendingPrompt = new();

    protected OnnxInferenceChatTemplateGenaiEngine(
        AssistantConfig config,
        string engineId,
        ILogger? logger,
        IReadOnlyList<string>? additionalStopTokens = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        EngineId = engineId;
        _logger = logger;

        var modelPath = config.ResolvedGenaiModelPath;
        if (string.IsNullOrWhiteSpace(modelPath) || !Directory.Exists(modelPath))
        {
            throw new DirectoryNotFoundException($"GenAI model directory not found: {modelPath}");
        }

        var templatePath = Path.Combine(modelPath, "chat_template.jinja");
        if (!File.Exists(templatePath))
        {
            throw new FileNotFoundException("GenAI model requires chat_template.jinja.", templatePath);
        }

        _chatTemplate = File.ReadAllText(templatePath);
        if (string.IsNullOrWhiteSpace(_chatTemplate))
        {
            throw new InvalidDataException("chat_template.jinja must not be empty.");
        }

        using var genaiConfig = new Config(modelPath);
        if (config.UseCuda)
        {
            try
            {
                genaiConfig.ClearProviders();
                genaiConfig.AppendProvider("cuda");
                DecoderUsingCuda = true;
                _logger?.LogInformation("{EngineId} configured with CUDA provider.", EngineId);
            }
            catch (Exception exception)
            {
                _logger?.LogWarning(
                    exception,
                    "{EngineId} CUDA provider unavailable; using the default provider.",
                    EngineId);
            }
        }

        _model = new Model(genaiConfig);
        _tokenizer = new Tokenizer(_model);
        LoadStopTokenIds(modelPath);
        if (additionalStopTokens is not null)
        {
            foreach (var stopToken in additionalStopTokens)
            {
                AddStopToken(stopToken);
            }
        }
    }

    public string EngineId { get; }
    public long HistoryLength { get; private set; }
    public bool DecoderUsingCuda { get; }
    public bool EmbeddingUsingCuda => false;

    // ======================= 可覆盖点 =======================

    /// <summary>生成起点前缀。非思考模式下可用于结构性关闭思考（例如 DeepSeek-R1 预填充空思考块）。</summary>
    protected virtual string GetGenerationPrefix(GenerationConfig config) => string.Empty;

    /// <summary>格式化后的 prompt 最终修补。默认原样返回。</summary>
    protected virtual string ApplyThinkingMode(string prompt, bool thinkMode)
    {
        _ = thinkMode;
        return prompt;
    }

    /// <summary>对“当前用户问题”做适配（历史消息不经过此方法）。默认原样返回。</summary>
    protected virtual string AdaptUserMessage(string content, bool thinkMode)
    {
        _ = thinkMode;
        return content;
    }

    /// <summary>对最终生成文本做归一化（例如剥离残留的思考标签）。默认原样返回。</summary>
    protected virtual string NormalizeGeneratedText(string text, GenerationConfig config)
    {
        _ = config;
        return text;
    }

    /// <summary>对 tokenizer 流式解码出的单个文本片段做归一化。</summary>
    protected virtual string NormalizeDecodedChunk(string text) => text;

    // ======================= Prompt 格式化 =======================

    public string FormatSystemPrompt(string systemPrompt) =>
        FormatMessages([new ChatMessageDto("system", systemPrompt)], addGenerationPrompt: false);

    public string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns)
    {
        var messages = new List<ChatMessageDto>(turns.Count * 2);
        foreach (var turn in turns)
        {
            messages.Add(new ChatMessageDto("user", turn.User));
            if (turn.Assistant.Length > 0)
            {
                messages.Add(new ChatMessageDto("assistant", turn.Assistant));
            }
        }
        return FormatMessages(messages, addGenerationPrompt: false);
    }

    public virtual string FormatUserPrompt(string question, bool thinkMode) =>
        ApplyThinkingMode(
            FormatMessages(
                [new ChatMessageDto("user", AdaptUserMessage(question, thinkMode))],
                addGenerationPrompt: true),
            thinkMode);

    public string FormatConversationPrompt(
        string systemPrompt,
        IReadOnlyList<ChatTurn> history,
        string question,
        bool thinkMode)
    {
        var messages = new List<ChatMessageDto>(history.Count * 2 + 2);
        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            messages.Add(new ChatMessageDto("system", systemPrompt));
        }
        foreach (var turn in history)
        {
            messages.Add(new ChatMessageDto("user", turn.User));
            if (turn.Assistant.Length > 0)
            {
                messages.Add(new ChatMessageDto("assistant", turn.Assistant));
            }
        }
        messages.Add(new ChatMessageDto("user", AdaptUserMessage(question, thinkMode)));
        return ApplyThinkingMode(
            FormatMessages(messages, addGenerationPrompt: true),
            thinkMode);
    }

    // ======================= Token 计数 / 状态 =======================

    public int CountTokens(string text)
    {
        using var sequences = _tokenizer.Encode(text);
        return sequences.NumSequences == 0 ? 0 : sequences[0].Length;
    }

    public void ResetState()
    {
        _pendingPrompt.Clear();
        HistoryLength = 0;
    }

    public void Prefill(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        _pendingPrompt.Append(text);
        HistoryLength = CountTokens(_pendingPrompt.ToString());
    }

    // ======================= 生成 =======================

    public (string Text, GenerationStats Stats) Generate(
        string prompt,
        GenerationConfig config,
        Action<string>? onToken = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(config);

        var fullPrompt = _pendingPrompt.Append(prompt).ToString();
        _pendingPrompt.Clear();
        HistoryLength = 0;

        using var sequences = _tokenizer.Encode(fullPrompt);
        var inputTokenCount = sequences.NumSequences == 0 ? 0 : sequences[0].Length;
        if (inputTokenCount >= _config.MaxSequenceLength)
        {
            throw new InvalidOperationException(
                $"Prompt exceeds the configured maximum sequence length ({_config.MaxSequenceLength}).");
        }

        using var generatorParams = new GeneratorParams(_model);
        var maxLength = Math.Min(inputTokenCount + config.MaxNewTokens, _config.MaxSequenceLength);
        generatorParams.SetSearchOption("max_length", maxLength);
        var maximumGeneratedTokens = maxLength - inputTokenCount;
        generatorParams.SetSearchOption("temperature", config.Temperature);
        generatorParams.SetSearchOption("top_p", config.TopP);
        generatorParams.SetSearchOption("top_k", config.TopK);
        generatorParams.SetSearchOption("do_sample", config.DoSample);
        generatorParams.SetSearchOption("repetition_penalty", config.RepetitionPenalty);

        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokenSequences(sequences);
        using var tokenizerStream = _tokenizer.CreateStream();

        // 记录序列初始长度（prompt token 数），用于 EOS 判定
        var previousSequenceLength = generator.GetSequence(0).Length;

        var generationPrefix = GetGenerationPrefix(config);
        var generatedText = new System.Text.StringBuilder(generationPrefix);
        if (config.Stream && generationPrefix.Length > 0)
        {
            onToken?.Invoke(generationPrefix);
        }

        var tokenIds = new List<int>();
        var wasCancelled = false;
        var loopDetected = false;
        var stopReason = string.Empty;
        var sw = Stopwatch.StartNew();

        while (!generator.IsDone())
        {
            if (cancellationToken.IsCancellationRequested)
            {
                wasCancelled = true;
                stopReason = "用户停止生成";
                break;
            }

            generator.GenerateNextToken();
            var outputTokens = generator.GetSequence(0);

            // 关键修复：GenerateNextToken 产出 EOS 时，ONNX Runtime GenAI 不会把 EOS 追加到序列，
            // 序列长度保持不变。此时若继续取 [^1] 会重复输出上一个可见 token。
            if (outputTokens.Length <= previousSequenceLength)
            {
                stopReason = "遇到 EOS token";
                break;
            }
            previousSequenceLength = outputTokens.Length;

            var tokenId = checked((int)outputTokens[^1]);
            tokenIds.Add(tokenId);
            var decoded = NormalizeDecodedChunk(tokenizerStream.Decode(tokenId));
            if (_stopTokenIds.Contains(tokenId))
            {
                stopReason = $"遇到停止 token ({tokenId})";
                break;
            }
            if (decoded.Length > 0)
            {
                generatedText.Append(decoded);
                if (config.Stream)
                {
                    onToken?.Invoke(decoded);
                }
            }

            if (config.EnableLoopDetection &&
                HasRepeatedTokenPattern(tokenIds, config.LoopCheckWindow, config.LoopRepeatThreshold))
            {
                loopDetected = true;
                stopReason = "检测到重复输出循环";
                _logger?.LogWarning(
                    "{EngineId} generation stopped after detecting a repeated token loop at {TokenCount} tokens.",
                    EngineId,
                    tokenIds.Count);
                break;
            }
        }

        if (string.IsNullOrEmpty(stopReason))
        {
            if (tokenIds.Count >= maximumGeneratedTokens)
            {
                stopReason = maxLength < inputTokenCount + config.MaxNewTokens
                    ? "达到模型最大上下文长度"
                    : "达到最大生成长度";
            }
            else
            {
                stopReason = "模型生成器结束（未检测到显式 EOS）";
            }
        }

        wasCancelled |= cancellationToken.IsCancellationRequested;
        if (wasCancelled)
        {
            stopReason = "用户停止生成";
        }
        sw.Stop();
        var seconds = sw.Elapsed.TotalSeconds;

        var finalText = NormalizeGeneratedText(generatedText.ToString(), config);

        return (finalText, new GenerationStats(
            tokenIds.Count,
            seconds,
            seconds > 0 ? tokenIds.Count / seconds : 0,
            tokenIds.Count > 0 ? seconds / tokenIds.Count : 0,
            loopDetected,
            false,
            wasCancelled,
            stopReason));
    }

    public void Dispose()
    {
        _tokenizer.Dispose();
        _model.Dispose();
    }

    // ======================= 内部工具 =======================

    private string FormatMessages(IReadOnlyList<ChatMessageDto> messages, bool addGenerationPrompt)
    {
        var messagesJson = JsonSerializer.Serialize(
            messages.ToList(), ChatJsonContext.Default.ListChatMessageDto);
        return _tokenizer.ApplyChatTemplate(
            _chatTemplate, messagesJson, string.Empty, addGenerationPrompt);
    }

    private void LoadStopTokenIds(string modelPath)
    {
        var genaiConfigPath = Path.Combine(modelPath, "genai_config.json");
        if (File.Exists(genaiConfigPath))
        {
            using var configDocument = JsonDocument.Parse(File.ReadAllText(genaiConfigPath));
            if (configDocument.RootElement.TryGetProperty("model", out var model) &&
                model.TryGetProperty("eos_token_id", out var eosTokenIds))
            {
                if (eosTokenIds.ValueKind == JsonValueKind.Number)
                {
                    _stopTokenIds.Add(eosTokenIds.GetInt32());
                }
                else if (eosTokenIds.ValueKind == JsonValueKind.Array)
                {
                    foreach (var tokenId in eosTokenIds.EnumerateArray())
                    {
                        if (tokenId.ValueKind == JsonValueKind.Number)
                        {
                            _stopTokenIds.Add(tokenId.GetInt32());
                        }
                    }
                }
            }
        }

        AddEosTokenFromTokenizerConfig(modelPath);
        if (_stopTokenIds.Count == 0)
        {
            _logger?.LogWarning(
                "{EngineId} has no explicit EOS token id; generation will rely on GenAI Generator.IsDone().",
                EngineId);
        }
    }

    private void AddEosTokenFromTokenizerConfig(string modelPath)
    {
        var tokenizerConfigPath = Path.Combine(modelPath, "tokenizer_config.json");
        if (!File.Exists(tokenizerConfigPath))
        {
            return;
        }

        using var tokenizerConfig = JsonDocument.Parse(File.ReadAllText(tokenizerConfigPath));
        if (!tokenizerConfig.RootElement.TryGetProperty("eos_token", out var eosToken))
        {
            return;
        }

        var value = eosToken.ValueKind switch
        {
            JsonValueKind.String => eosToken.GetString(),
            JsonValueKind.Object when eosToken.TryGetProperty("content", out var content) &&
                                     content.ValueKind == JsonValueKind.String => content.GetString(),
            _ => null
        };
        if (string.IsNullOrWhiteSpace(value))
        {
            return;
        }

        using var encoded = _tokenizer.Encode(value);
        if (encoded.NumSequences == 1 && encoded[0].Length == 1)
        {
            _stopTokenIds.Add(checked((int)encoded[0][0]));
        }
    }

    private void AddStopToken(string token)
    {
        using var encoded = _tokenizer.Encode(token);
        if (encoded.NumSequences == 1 && encoded[0].Length == 1)
        {
            _stopTokenIds.Add(checked((int)encoded[0][0]));
            _logger?.LogInformation(
                "{EngineId} additional stop token {Token} resolved to ID {TokenId}.",
                EngineId,
                token,
                encoded[0][0]);
            return;
        }

        _logger?.LogWarning(
            "{EngineId} additional stop token {Token} did not encode as one token and will not be used.",
            EngineId,
            token);
    }

    private static bool HasRepeatedTokenPattern(
        IReadOnlyList<int> tokens,
        int window,
        int repeatThreshold)
    {
        if (window <= 0 || repeatThreshold <= 1 || tokens.Count < window)
        {
            return false;
        }

        var start = tokens.Count - window;
        for (var patternLength = 3; patternLength <= 9; patternLength++)
        {
            if (window < patternLength * repeatThreshold)
            {
                continue;
            }

            var repetitions = 1;
            var cursor = tokens.Count - patternLength * 2;
            var lastStart = tokens.Count - patternLength;
            while (cursor >= start && repetitions < repeatThreshold)
            {
                var matches = true;
                for (var offset = 0; offset < patternLength; offset++)
                {
                    if (tokens[cursor + offset] != tokens[lastStart + offset])
                    {
                        matches = false;
                        break;
                    }
                }
                if (!matches)
                {
                    break;
                }
                repetitions++;
                cursor -= patternLength;
            }

            if (repetitions >= repeatThreshold)
            {
                return true;
            }
        }
        return false;
    }
}