using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using Tokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer;

namespace Onar.Inference;

/// <summary>
/// 供 ApplyChatTemplate 使用的消息 DTO。
/// 显式使用属性 + [JsonPropertyName]，避免依赖匿名类型的反射序列化。
/// </summary>
internal sealed record ChatMessageDto(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

/// <summary>
/// System.Text.Json 源生成器上下文。启用 trimming/AOT 后必须使用此上下文序列化。
/// </summary>
[JsonSerializable(typeof(List<ChatMessageDto>))]
internal partial class ChatJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Llama 3.x ONNX Runtime GenAI 推理引擎。
/// Uses an optional tokenizer chat template or the standard Llama 3.x message format.
///
/// 本版本针对"最后一个 token 重复"（如 1010、two!!）做了三层防护：
///   1. token 级相邻去重：连续两个相同非 structural token 视为答案重复，直接停止；
///   2. tokenizerStream 每 token 必喂：避免 structural token 被 continue 跳过导致字节缓冲错位；
///   3. 文本级尾缀去重兜底：覆盖 10 = 1 + 0 两个 token 的场景。
/// </summary>
public sealed class OnnxInferenceLlamaGenaiEngine :
    IInferenceEngine, IConversationPromptFormatter, IDisposable
{
    private const string BeginOfText = "<|begin_of_text|>";
    private const string StartHeader = "<|start_header_id|>";
    private const string EndHeader = "<|end_header_id|>";
    private const string EndOfTurn = "<|eot_id|>";

    /// <summary>短答案 token 级去重阈值：MaxNewTokens ≤ 此值时启用，避免误杀长文本。</summary>
    private const int AggressiveDedupMaxTokens = 512;

    /// <summary>环境变量 LLAMA_DISABLE_DEDUP=1 可关闭尾缀去重（调试用）。</summary>
    private static readonly bool DisableDedup =
        string.Equals(
            Environment.GetEnvironmentVariable("LLAMA_DISABLE_DEDUP"),
            "1", StringComparison.OrdinalIgnoreCase);

    private readonly AssistantConfig _config;
    private readonly ILogger<OnnxInferenceLlamaGenaiEngine>? _logger;
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly string? _chatTemplate;
    private readonly HashSet<int> _structuralTokenIds;
    private readonly HashSet<int> _stopTokenIds;
    private readonly System.Text.StringBuilder _pendingPrompt = new();

    public long HistoryLength { get; private set; }
    public string EngineId => "llama-onnx-genai";
    public bool DecoderUsingCuda { get; }
    public bool EmbeddingUsingCuda => false;

    public OnnxInferenceLlamaGenaiEngine(
        AssistantConfig config,
        ILogger<OnnxInferenceLlamaGenaiEngine>? logger = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger;

        var modelPath = config.ResolvedGenaiModelPath;
        if (string.IsNullOrEmpty(modelPath) || !Directory.Exists(modelPath))
            throw new DirectoryNotFoundException($"GenAI model directory not found: {modelPath}");

        using var genaiConfig = new Config(modelPath);
        if (config.UseCuda)
        {
            try
            {
                genaiConfig.ClearProviders();
                genaiConfig.AppendProvider("cuda");
                DecoderUsingCuda = true;
                _logger?.LogInformation("GenAI model configured with CUDA provider.");
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "CUDA provider unavailable, falling back to default provider.");
                DecoderUsingCuda = false;
            }
        }
        else
        {
            DecoderUsingCuda = false;
        }

        _model = new Model(genaiConfig);
        _tokenizer = new Tokenizer(_model);

        var tokenizerConfigPath = Path.Combine(modelPath, "tokenizer_config.json");
        if (!File.Exists(tokenizerConfigPath))
        {
            Dispose();
            throw new FileNotFoundException(
                "Llama GenAI requires tokenizer_config.json.",
                tokenizerConfigPath);
        }

        try
        {
            using var tokenizerConfig = JsonDocument.Parse(File.ReadAllText(tokenizerConfigPath));
            if (tokenizerConfig.RootElement.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("tokenizer_config.json must contain a JSON object.");
            }

            var templatePath = Path.Combine(modelPath, "chat_template.jinja");
            if (File.Exists(templatePath))
            {
                _chatTemplate = File.ReadAllText(templatePath);
                if (string.IsNullOrWhiteSpace(_chatTemplate))
                {
                    throw new InvalidDataException(
                        "chat_template.jinja must not be empty.");
                }
            }
            else if (tokenizerConfig.RootElement.TryGetProperty("chat_template", out var chatTemplate))
            {
                if (chatTemplate.ValueKind != JsonValueKind.String ||
                    string.IsNullOrWhiteSpace(chatTemplate.GetString()))
                {
                    throw new InvalidDataException(
                        "tokenizer_config.json chat_template must be a non-empty string when present.");
                }
                _chatTemplate = chatTemplate.GetString();
            }
        }
        catch
        {
            Dispose();
            throw;
        }

        var endOfTurnTokenId = GetSpecialTokenId(EndOfTurn);
        var endOfTextTokenId = GetSpecialTokenId("<|end_of_text|>");
        _structuralTokenIds = new HashSet<int>
        {
            GetSpecialTokenId(BeginOfText),
            GetSpecialTokenId(StartHeader),
            GetSpecialTokenId(EndHeader),
            endOfTurnTokenId,
            endOfTextTokenId
        };
        _stopTokenIds = [endOfTurnTokenId, endOfTextTokenId];
    }

    public string FormatSystemPrompt(string systemPrompt) =>
        ApplyChatTemplate([new ChatMessage("system", systemPrompt)], addGenerationPrompt: false);

    public string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns)
    {
        var messages = new List<ChatMessage>(turns.Count * 2);
        foreach (var turn in turns)
        {
            messages.Add(new ChatMessage("user", turn.User));
            if (turn.Assistant.Length > 0)
            {
                messages.Add(new ChatMessage("assistant", turn.Assistant));
            }
        }
        return ApplyChatTemplate(messages, addGenerationPrompt: false);
    }

    public string FormatUserPrompt(string question, bool thinkMode)
    {
        _ = thinkMode;
        return ApplyChatTemplate([new ChatMessage("user", question)], addGenerationPrompt: true);
    }

    public string FormatConversationPrompt(
        string systemPrompt,
        IReadOnlyList<ChatTurn> history,
        string question,
        bool thinkMode)
    {
        _ = thinkMode;
        var messages = new List<ChatMessage>(1 + history.Count * 2 + 1);
        if (!string.IsNullOrWhiteSpace(systemPrompt))
        {
            messages.Add(new ChatMessage("system", systemPrompt));
        }
        foreach (var turn in history)
        {
            messages.Add(new ChatMessage("user", turn.User));
            if (turn.Assistant.Length > 0)
            {
                messages.Add(new ChatMessage("assistant", turn.Assistant));
            }
        }
        messages.Add(new ChatMessage("user", question));
        return ApplyChatTemplate(messages, addGenerationPrompt: true);
    }

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

    private int GetSpecialTokenId(string token)
    {
        using var encoded = _tokenizer.Encode(token);
        if (encoded.NumSequences != 1 || encoded[0].Length != 1)
        {
            throw new InvalidDataException(
                $"Llama tokenizer did not encode {token} as one special token.");
        }
        return checked((int)encoded[0][0]);
    }

    private string ApplyChatTemplate(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt)
    {
        if (_chatTemplate is null)
        {
            var prompt = new System.Text.StringBuilder(BeginOfText);
            foreach (var message in messages)
            {
                prompt.Append(StartHeader)
                    .Append(message.Role)
                    .Append(EndHeader)
                    .Append("\n\n")
                    .Append(message.Content)
                    .Append(EndOfTurn);
            }

            if (addGenerationPrompt)
            {
                prompt.Append(StartHeader)
                    .Append("assistant")
                    .Append(EndHeader)
                    .Append("\n\n");
            }
            return prompt.ToString();
        }

        var messagesJson = JsonSerializer.Serialize(messages.Select(m => new ChatMessageDto(m.Role, m.Content)).ToList(), ChatJsonContext.Default.ListChatMessageDto);
        return _tokenizer.ApplyChatTemplate(
            _chatTemplate, messagesJson, string.Empty, addGenerationPrompt);
    }

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

        var text = new System.Text.StringBuilder();
        var generatedTokens = new List<int>();
        var tokenCount = 0;
        var loopDetected = false;
        var dedupStopped = false;
        var wasCancelled = false;
        var stopReason = string.Empty;
        var sw = Stopwatch.StartNew();

        // 短答案 token 级去重开关：长文本生成时关闭，避免误杀自然重复（如 "\n\n"、"。."）
        var aggressiveDedup = config.MaxNewTokens <= AggressiveDedupMaxTokens;
        if (aggressiveDedup)
        {
            _logger?.LogDebug(
                "Token-level dedup enabled (MaxNewTokens={Max}).", config.MaxNewTokens);
        }

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
            if (outputTokens.Length == 0) continue;

            var tokenId = checked((int)outputTokens[^1]);

            // ★ 修复 A：token 级相邻去重
            //   连续两个相同非 structural token 视为答案重复（如 "10" "10" → 只保留一个）
            //   structural token（EOS、header 等）排除，避免误判
            if (aggressiveDedup &&
                generatedTokens.Count >= 1 &&
                generatedTokens[^1] == tokenId &&
                !_structuralTokenIds.Contains(tokenId))
            {
                _logger?.LogWarning(
                    "Duplicate trailing token detected: id={Id} at position {Pos}. " +
                    "Dropping duplicate and stopping.",
                    tokenId, generatedTokens.Count + 1);
                dedupStopped = true;
                stopReason = "检测到连续重复 token";
                break;
            }

            generatedTokens.Add(tokenId);
            tokenCount++;

            // ★ 修复 B：每个 token 都必须喂给 tokenizerStream（包括 structural token）
            //   否则 byte-level BPE 的内部字节缓冲会错位，后续解码可能断裂
            var tokenText = tokenizerStream.Decode(tokenId);

            if (_structuralTokenIds.Contains(tokenId))
            {
                if (_stopTokenIds.Contains(tokenId))
                {
                    stopReason = $"遇到 EOS token ({tokenId})";
                    break;
                }
                continue;
            }

            if (tokenText.Length == 0)
            {
                continue;
            }

            text.Append(tokenText);
            if (config.Stream)
            {
                onToken?.Invoke(tokenText);
            }

            if (config.EnableLoopDetection &&
                DetectLoop(generatedTokens, config.LoopCheckWindow, config.LoopRepeatThreshold))
            {
                loopDetected = true;
                stopReason = "检测到重复输出循环";
                _logger?.LogWarning(
                    "Llama generation stopped after detecting a repeated token loop at {TokenCount} tokens",
                    tokenCount);
                break;
            }
        }

        if (string.IsNullOrEmpty(stopReason))
        {
            stopReason = tokenCount >= maximumGeneratedTokens
                ? (maxLength < inputTokenCount + config.MaxNewTokens
                    ? "达到模型最大上下文长度"
                    : "达到最大生成长度")
                : "模型生成器结束（未检测到显式 EOS）";
        }

        wasCancelled |= cancellationToken.IsCancellationRequested;
        if (wasCancelled)
        {
            stopReason = "用户停止生成";
        }
        sw.Stop();

        // ★ 修复 C：文本级尾缀去重兜底
        //   覆盖 token 级抓不到的 case（如 "10" 被拆成 "1" "0" 两个 token，
        //   模型又生成一遍 "1" "0"，相邻 token 不相同但文本末尾是 "1010"）
        var finalText = text.ToString();
        if (!DisableDedup && !wasCancelled && !loopDetected)
        {
            var deduped = DeduplicateTrailingSuffix(finalText);
            if (!string.Equals(finalText, deduped, StringComparison.Ordinal))
            {
                _logger?.LogWarning(
                    "Trailing suffix dedup applied: '{Before}' -> '{After}'",
                    finalText.Length > 30 ? "..." + finalText[^30..] : finalText,
                    deduped.Length > 30 ? "..." + deduped[^30..] : deduped);
                finalText = deduped;
            }
        }

        var totalSeconds = sw.Elapsed.TotalSeconds;
        return (finalText, new GenerationStats(
            tokenCount,
            totalSeconds,
            totalSeconds > 0 ? tokenCount / totalSeconds : 0,
            tokenCount > 0 ? totalSeconds / tokenCount : 0,
            loopDetected || dedupStopped,
            false,
            wasCancelled,
            stopReason));
    }

    /// <summary>
    /// 文本级尾缀去重：检测输出末尾是否有重复片段，去掉一份。
    ///   "1010"   → "10"   （len=2，"10"+"10"）
    ///   "two!!"  → "two!" （len=1，"!"+"!"）
    ///   "100100" → "100"  （len=3）
    ///   "three"  → "three"（不重复，原样返回）
    /// </summary>
    private static string DeduplicateTrailingSuffix(string text)
    {
        if (text.Length < 2) return text;

        // 从 1 到 6 个字符的尾缀长度尝试
        for (var len = 1; len <= 6 && len * 2 <= text.Length; len++)
        {
            var tail = text.Substring(text.Length - len);
            var beforeTail = text.Substring(text.Length - len * 2, len);
            if (string.Equals(tail, beforeTail, StringComparison.Ordinal))
            {
                return text.Substring(0, text.Length - len);
            }
        }
        return text;
    }

    private static bool DetectLoop(IReadOnlyList<int> tokens, int window, int repeatThreshold)
    {
        if (window <= 0 || repeatThreshold <= 0 || tokens.Count < window)
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

            var repeats = 1;
            var cursor = tokens.Count - patternLength * 2;
            var lastStart = tokens.Count - patternLength;
            while (cursor >= start && repeats < repeatThreshold)
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

                repeats++;
                cursor -= patternLength;
            }

            if (repeats >= repeatThreshold)
            {
                return true;
            }
        }

        return false;
    }

    public void Dispose()
    {
        _tokenizer?.Dispose();
        _model?.Dispose();
    }

    private sealed record ChatMessage(string Role, string Content);
}