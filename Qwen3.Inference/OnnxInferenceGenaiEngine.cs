using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.Diagnostics;
using System.Text.Json;
using Tokenizer = Microsoft.ML.OnnxRuntimeGenAI.Tokenizer;

namespace Qwen3.Inference;

/// <summary>
/// Qwen3 ONNX Runtime GenAI 推理引擎。
///
/// 本版本针对"最后一个 token 重复"（如 22、1010）做了修复：
///   1. _stopTokenIds 强制包含 Qwen3 的 151645(&lt;|im_end|&gt;) 和 151643(&lt;|endoftext|&gt;)，
///      不依赖 genai_config.json 是否完整；
///   2. 停止 token 主动 break，不依赖 IsDone()；
///   3. 累积 decode 替代逐 token decode，保证 byte-level 多字节安全；
///   4. token 级相邻去重：连续两个相同非 structural token 视为答案重复，直接停止；
///   5. 文本级尾缀去重兜底：覆盖 10 = 1 + 0 两个 token 的场景。
///
/// 注意：FormatUserPrompt 的 thinkMode 逻辑保持原样——
///   thinkMode=true  → 不加空 &lt;think&gt;&lt;/think&gt;，让模型自发生成思考链；
///   thinkMode=false → 加空 &lt;think&gt;\n\n&lt;/think&gt;\n\n，跳过思考直接回答。
/// </summary>
public sealed class OnnxInferenceGenaiEngine : IInferenceEngine, IDisposable
{
    // Qwen3 特殊 token 字面量
    private const string ImStart = "<|im_start|>";
    private const string ImEnd = "<|im_end|>";
    private const string EndOfText = "<|endoftext|>";

    // Qwen3 特殊 token id（Qwen3 系列固定）
    private const int QwenImEndId = 151645;
    private const int QwenEndOfTextId = 151643;
    private const int QwenImStartId = 151644;

    private static readonly string[] SpecialTokenLiterals =
    {
        ImStart, ImEnd, EndOfText,
    };

    /// <summary>短答案 token 级去重阈值：MaxNewTokens ≤ 此值时启用，避免误杀长文本。</summary>
    private const int AggressiveDedupMaxTokens = 512;

    /// <summary>环境变量 QWEN_DISABLE_DEDUP=1 可关闭尾缀去重（调试用）。</summary>
    private static readonly bool DisableDedup =
        string.Equals(
            Environment.GetEnvironmentVariable("QWEN_DISABLE_DEDUP"),
            "1", StringComparison.OrdinalIgnoreCase);

    private readonly AssistantConfig _config;
    private readonly ILogger<OnnxInferenceGenaiEngine>? _logger;
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;
    private readonly HashSet<int> _structuralTokenIds;
    private readonly HashSet<int> _stopTokenIds;
    private readonly System.Text.StringBuilder _pendingPrompt = new();

    public long HistoryLength { get; private set; }
    public string EngineId => "qwen3-onnx-genai";
    public bool DecoderUsingCuda { get; }
    public bool EmbeddingUsingCuda => false;

    public OnnxInferenceGenaiEngine(AssistantConfig config, ILogger<OnnxInferenceGenaiEngine>? logger = null)
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

        // 结构性 token：不输出但继续生成（本引擎里全部作停止，因为对话场景不应生成 im_start）
        _structuralTokenIds = new HashSet<int>
        {
            QwenImStartId, QwenImEndId, QwenEndOfTextId,
        };

        // 停止 token：硬编码保底 + 从 genai_config.json 补齐
        _stopTokenIds = new HashSet<int>
        {
            QwenImStartId, QwenImEndId, QwenEndOfTextId,
        };
        LoadEosTokenIds(modelPath, _stopTokenIds);

        _logger?.LogInformation(
            "Qwen structural token ids: {StructIds}; stop token ids: {StopIds}",
            string.Join(",", _structuralTokenIds),
            string.Join(",", _stopTokenIds));

        if (_stopTokenIds.Count == 0)
        {
            _logger?.LogWarning(
                "No stop token ids resolved; relying on max_length only.");
        }
    }

    // ======================= Prompt 构建 =======================

    public string FormatSystemPrompt(string systemPrompt) =>
        $"<|im_start|>system\n{systemPrompt}<|im_end|>\n";

    public string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var turn in turns)
        {
            sb.Append("<|im_start|>user\n").Append(turn.User).Append("<|im_end|>\n");
            if (turn.Assistant.Length > 0)
                sb.Append("<|im_start|>assistant\n").Append(turn.Assistant).Append("<|im_end|>\n");
        }
        return sb.ToString();
    }

    public string FormatUserPrompt(string question, bool thinkMode)
    {
        var prompt = $"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n";
        // thinkMode=true  → 不加空 think，让模型自发生成 <think>...</think> 再回答
        // thinkMode=false → 加空 <think></think>，让模型跳过思考直接回答
        return thinkMode ? prompt : prompt + "<think>\n\n</think>\n\n";
    }

    // ======================= Token 统计 / 状态 =======================

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

    // ======================= 生成主循环 =======================

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

        _logger?.LogInformation("Qwen prompt token count: {Count}", inputTokenCount);
        _logger?.LogDebug("Qwen prompt raw: {Prompt}", fullPrompt);

        // ---------- 采样参数 ----------
        using var generatorParams = new GeneratorParams(_model);
        var maxLength = Math.Min(inputTokenCount + config.MaxNewTokens, _config.MaxSequenceLength);
        generatorParams.SetSearchOption("max_length", maxLength);
        var maximumGeneratedTokens = maxLength - inputTokenCount;
        generatorParams.SetSearchOption("temperature", config.Temperature);
        generatorParams.SetSearchOption("top_p", config.TopP);
        generatorParams.SetSearchOption("top_k", config.TopK);
        generatorParams.SetSearchOption("do_sample", config.DoSample);
        generatorParams.SetSearchOption("repetition_penalty", config.RepetitionPenalty);

        _logger?.LogDebug(
            "Sampling: do_sample={DoSample}, temp={Temp}, top_p={TopP}, top_k={TopK}, rep={Rep}, max_new={MaxNew}",
            config.DoSample, config.Temperature, config.TopP, config.TopK,
            config.RepetitionPenalty, config.MaxNewTokens);

        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokenSequences(sequences);

        var text = new System.Text.StringBuilder();
        // 累积解码
        var generatedTokens = new List<int>(Math.Max(config.MaxNewTokens, 64));
        var lastEmittedLength = 0;
        var tokenCount = 0;
        var wasCancelled = false;
        var hitStopToken = false;
        var dedupStopped = false;
        var stopReason = string.Empty;
        var sw = Stopwatch.StartNew();

        var aggressiveDedup = config.MaxNewTokens <= AggressiveDedupMaxTokens;
        var debugEnabled = _logger is not null && _logger.IsEnabled(LogLevel.Debug);

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

            if (debugEnabled)
            {
                var single = _tokenizer.Decode(new[] { tokenId });
                _logger!.LogDebug("gen[{Count}] id={Id} text='{Text}'",
                    tokenCount + 1, tokenId, single);
            }

            // ---------- 修复 1：token 级相邻去重 ----------
            //   连续两个相同非 structural token 视为答案重复（如 [2, 2] → 只保留一个）
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

            // ---------- 修复 2：停止 token 主动 break ----------
            if (_stopTokenIds.Contains(tokenId))
            {
                hitStopToken = true;
                stopReason = $"遇到 EOS token ({tokenId})";
                _logger?.LogDebug("Stop token hit: id={Id} at {Count} tokens", tokenId, tokenCount);
                break;
            }

            if (_structuralTokenIds.Contains(tokenId))
            {
                continue;
            }

            // ---------- 修复 3：累积 decode ----------
            //   每次 decode 完整序列，取相对上次的增量；特殊 token 从结果剔除
            var fullText = StripSpecialTokens(_tokenizer.Decode(generatedTokens.ToArray()));
            if (fullText.Length > lastEmittedLength)
            {
                var delta = fullText.Substring(lastEmittedLength);
                lastEmittedLength = fullText.Length;
                text.Append(delta);
                if (config.Stream)
                {
                    onToken?.Invoke(delta);
                }
            }
        }

        if (string.IsNullOrEmpty(stopReason))
        {
            stopReason = generatedTokens.Count >= maximumGeneratedTokens
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

        // ---------- 修复 4：文本级尾缀去重兜底 ----------
        //   覆盖 token 级抓不到的 case（如 "10" 被拆成 "1" "0" 两个 token，模型又生成一遍）
        var finalText = text.ToString();
        if (!DisableDedup && !wasCancelled)
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
            dedupStopped,
            hitStopToken,
            wasCancelled,
            stopReason));
    }

    // ======================= 辅助 =======================

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

    private static string StripSpecialTokens(string text)
    {
        if (text.Length == 0) return text;
        if (text.IndexOf('<') < 0) return text;

        foreach (var token in SpecialTokenLiterals)
        {
            if (text.Contains(token, StringComparison.Ordinal))
            {
                text = text.Replace(token, string.Empty, StringComparison.Ordinal);
            }
        }
        return text;
    }

    private void LoadEosTokenIds(string modelPath, HashSet<int> target)
    {
        var genaiConfigPath = Path.Combine(modelPath, "genai_config.json");
        if (!File.Exists(genaiConfigPath))
        {
            _logger?.LogWarning("genai_config.json not found; cannot resolve eos_token_id.");
            return;
        }

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(genaiConfigPath));
            if (!doc.RootElement.TryGetProperty("model", out var model) ||
                !model.TryGetProperty("eos_token_id", out var eos))
            {
                return;
            }

            if (eos.ValueKind == JsonValueKind.Number)
            {
                target.Add(eos.GetInt32());
            }
            else if (eos.ValueKind == JsonValueKind.Array)
            {
                foreach (var e in eos.EnumerateArray())
                {
                    if (e.ValueKind == JsonValueKind.Number)
                    {
                        target.Add(e.GetInt32());
                    }
                }
            }
            _logger?.LogInformation(
                "eos_token_id from genai_config.json: {Ids}",
                string.Join(",", target));
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Failed to parse genai_config.json for eos_token_id.");
        }
    }

    public void Dispose()
    {
        _tokenizer?.Dispose();
        _model?.Dispose();
    }
}