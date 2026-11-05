using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using Tokenizers.DotNet;

namespace Onar.Inference;

public sealed record GenerationStats(
    int TotalTokens,
    double TotalTimeSeconds,
    double TokensPerSecond,
    double AverageTimePerTokenSeconds,
    bool LoopDetected,
    bool ThinkingBudgetExceeded,
    bool WasCancelled = false,
    string StopReason = "未知");

public sealed class OnnxInferenceEngine : IInferenceEngine
{
    private const int ImEndTokenId = 151643;
    private const int EndOfTextTokenId = 151645;

    private readonly AssistantConfig _config;
    private readonly ILogger<OnnxInferenceEngine>? _logger;
    private readonly Tokenizer _tokenizer;
    private readonly InferenceSession _embeddingSession;
    private readonly InferenceSession _decoderSession;
    private readonly string _embeddingInputName;
    private readonly string _embeddingOutputName;
    private readonly int _numLayers;
    private readonly int _numKeyValueHeads;
    private readonly int _headDim;
    private readonly string[] _decoderOutputNames;
    private readonly HashSet<string> _decoderInputNames;
    private readonly int? _thinkEndTokenId;
    private readonly OrtMemoryInfo _memoryInfo = OrtMemoryInfo.DefaultInstance;

    // ========== KV Cache 复用核心字段 ==========
    private IDisposableReadOnlyCollection<OrtValue>? _previousDecoderOutputs;
    private readonly OrtValue[] _emptyKeys;
    private readonly OrtValue[] _emptyValues;

    // 小张量输入复用 buffer（避免每步分配）
    private readonly long[] _historyLenBuffer = new long[1];
    private readonly long[] _idsLenBuffer = new long[1];
    private readonly sbyte[] _attnBuffer = new sbyte[1];

    private long _historyLength;
    private sbyte _attentionMask = 1;
    private List<int> _generatedTokens = [];
    private readonly HashSet<int> _generatedTokenSet = new();

    public long HistoryLength => _historyLength;
    public string EngineId => "qwen3-onnx";

    /// <summary>当前 Decoder 是否配置为使用 CUDA（基于 session 创建是否成功）。</summary>
    public bool DecoderUsingCuda { get; }

    /// <summary>当前 Embedding 是否使用 CUDA（固定 false，Embedding 始终 CPU）。</summary>
    public bool EmbeddingUsingCuda => false;

    public OnnxInferenceEngine(AssistantConfig config, ILogger<OnnxInferenceEngine>? logger = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger;
        ValidateFiles(config);

        using var modelJson = JsonDocument.Parse(File.ReadAllText(config.ModelConfigPath));
        var modelConfig = modelJson.RootElement;
        var hiddenSize = ReadRequiredInt(modelConfig, "hidden_size");
        var attentionHeads = ReadRequiredInt(modelConfig, "num_attention_heads");
        _numLayers = ReadRequiredInt(modelConfig, "num_hidden_layers");
        if (attentionHeads <= 0 || hiddenSize <= 0 || hiddenSize % attentionHeads != 0)
            throw new InvalidDataException("config.json contains invalid attention or hidden dimensions.");
        _numKeyValueHeads = ReadOptionalInt(modelConfig, "num_key_value_heads", attentionHeads);
        _headDim = ReadOptionalInt(modelConfig, "head_dim", hiddenSize / attentionHeads);
        if (_numLayers <= 0 || _numKeyValueHeads <= 0 || _headDim <= 0)
            throw new InvalidDataException("config.json contains invalid model dimensions.");

        _tokenizer = new Tokenizer(config.TokenizerPath);
        var thinkEndTokens = _tokenizer.Encode("</think>").ToArray();
        _thinkEndTokenId = thinkEndTokens.Length == 0 ? null : checked((int)thinkEndTokens[^1]);
        _logger?.LogInformation("Loading Qwen3 ONNX model from {ModelPath}", config.ModelPath);

        // ==================== Embedding：固定 CPU ====================
        _embeddingSession = CreateCpuSession(config.EmbeddingModelPath, config);
        _logger?.LogInformation("Embedding session created on CPU.");

        // ==================== Decoder：按路径和构建后端判断 ====================
        var isInt4Model = config.DecoderModelPath.Contains("int4", StringComparison.OrdinalIgnoreCase);
#if INFERENCE_BACKEND_GPU
        if (config.UseCuda && isInt4Model)
        {
            _logger?.LogInformation("INT4 model selected; trying CUDA first, fallback to CPU.");
            _decoderSession = CreateDecoderSessionWithCudaFallback(
                config.DecoderModelPath, config, out bool usingCuda);
            DecoderUsingCuda = usingCuda;
            _logger?.LogInformation(usingCuda
                ? "Decoder session created with CUDA provider."
                : "CUDA unavailable or failed; decoder session fell back to CPU.");
        }
        else
#endif
        {
            _logger?.LogInformation(isInt4Model
                ? "Using CPU decoder session because CUDA is disabled or not selected."
                : "Using CPU decoder session.");
            _decoderSession = CreateCpuSession(config.DecoderModelPath, config);
            DecoderUsingCuda = false;
        }

        var embeddingInputs = _embeddingSession.InputMetadata.Keys.ToArray();
        if (embeddingInputs.Length != 1)
        {
            Dispose();
            throw new InvalidDataException($"Embedding model must have exactly one input; found {embeddingInputs.Length}.");
        }
        _embeddingInputName = embeddingInputs[0];
        _embeddingOutputName = _embeddingSession.OutputMetadata.Keys.First();
        _decoderInputNames = _decoderSession.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal);
        _decoderOutputNames = _decoderSession.OutputMetadata.Keys.ToArray();

        try
        {
            ValidateDecoderContract();

            // 预分配空 KV Cache（仅用于首次 prefill 前的“无历史”场景）
            _emptyKeys = new OrtValue[_numLayers];
            _emptyValues = new OrtValue[_numLayers];
            for (var i = 0; i < _numLayers; i++)
            {
                _emptyKeys[i] = OrtValue.CreateTensorValueFromMemory<float>(
                    _memoryInfo,
                    new Memory<float>(Array.Empty<float>()),
                    new long[] { _numKeyValueHeads, 1, _headDim, 0 });
                _emptyValues[i] = OrtValue.CreateTensorValueFromMemory<float>(
                    _memoryInfo,
                    new Memory<float>(Array.Empty<float>()),
                    new long[] { _numKeyValueHeads, 1, 0, _headDim });
            }

            _historyLength = 0;
            _attentionMask = 1;
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // ==================== Session 创建辅助 ====================

    private static SessionOptions BuildBaseOptions(AssistantConfig config, bool verboseLogging = false)
    {
        var options = new SessionOptions
        {
            IntraOpNumThreads = config.IntraOpNumThreads,
            InterOpNumThreads = config.InterOpNumThreads,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        options.AddSessionConfigEntry("session.enable_quant_qdq_cleanup", "1");
        options.AddSessionConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
        if (verboseLogging)
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
        return options;
    }

    private static InferenceSession CreateCpuSession(string modelPath, AssistantConfig config)
    {
        using var options = BuildBaseOptions(config);
        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// 优先 CUDA，失败回退 CPU。
    /// 判断依据：AppendExecutionProvider_CUDA 是否抛异常 + InferenceSession 是否成功构造。
    /// C# API 无 GetProviders()，无法在构造后精确查询实际生效的 provider，
    /// 因此以“追加成功 + 构造成功”作为 CUDA 可用的标志。
    /// </summary>
    private InferenceSession CreateDecoderSessionWithCudaFallback(
        string modelPath, AssistantConfig config, out bool usingCuda)
    {
        usingCuda = false;

        try
        {
            var cudaOptions = BuildBaseOptions(config);
            try
            {
                cudaOptions.AppendExecutionProvider_CUDA(0);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "AppendExecutionProvider_CUDA failed; falling back to CPU.");
                cudaOptions.Dispose();
                return CreateCpuSession(modelPath, config);
            }

            InferenceSession session;
            try
            {
                session = new InferenceSession(modelPath, cudaOptions);
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Decoder session creation with CUDA failed; falling back to CPU.");
                cudaOptions.Dispose();
                return CreateCpuSession(modelPath, config);
            }
            finally
            {
                cudaOptions.Dispose();
            }

            usingCuda = true;
            return session;
        }
        catch (Exception ex)
        {
            _logger?.LogWarning(ex, "Unexpected CUDA initialization failure; falling back to CPU.");
            return CreateCpuSession(modelPath, config);
        }
    }

    /// <summary>
    /// 可选：开启 Verbose 日志重建 Decoder session，用于排查 CUDA 是否真的生效。
    /// 仅在调试时调用，发布版本不要调用。
    /// </summary>
    public void LogDecoderProviderInfo()
    {
        if (_logger == null) return;
        try
        {
            using var options = BuildBaseOptions(_config, verboseLogging: true);
            // 仅创建并立即释放，用于触发 provider 注册日志
            using var probe = new InferenceSession(_config.DecoderModelPath, options);
            _logger.LogInformation("Verbose logging probe session created for decoder provider inspection.");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to create verbose logging probe session.");
        }
    }

    public string FormatSystemPrompt(string systemPrompt) =>
        $"<|im_start|>system\n{systemPrompt}<|im_end|>\n";

    public string FormatHistoryPrompt(IReadOnlyList<ChatTurn> turns)
    {
        var prompt = new System.Text.StringBuilder();
        foreach (var turn in turns)
        {
            prompt.Append("<|im_start|>user\n").Append(turn.User).Append("<|im_end|>\n");
            if (turn.Assistant.Length > 0)
                prompt.Append("<|im_start|>assistant\n").Append(turn.Assistant).Append("<|im_end|>\n");
        }
        return prompt.ToString();
    }

    public string FormatUserPrompt(string question, bool thinkMode)
    {
        var prompt = $"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n";
        return thinkMode ? prompt : prompt + "<think>\n\n</think>\n\n";
    }

    public int CountTokens(string text) => _tokenizer.Encode(text).Count();

    public void ResetState()
    {
        _previousDecoderOutputs?.Dispose();
        _previousDecoderOutputs = null;
        _historyLength = 0;
        _attentionMask = 1;
        _generatedTokens.Clear();
        _generatedTokenSet.Clear();
    }

    public void Prefill(string text)
    {
        var ids = EncodeInput(text);
        if (ids.Length == 0) return;
        RunDecoder(GetEmbedding(ids));
    }

    public (string Text, GenerationStats Stats) Generate(
        string prompt,
        GenerationConfig config,
        Action<string>? onToken = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(config);
        ValidateGenerationConfig(config);
        var inputIds = EncodeInput(prompt);
        if (inputIds.Length == 0)
            throw new InvalidOperationException("The prompt produced no tokens.");
        if (_historyLength + inputIds.Length > _config.MaxSequenceLength)
            throw new InvalidOperationException(
                $"Prompt and KV cache exceed the configured maximum sequence length ({_config.MaxSequenceLength}).");

        if (inputIds.Length > 1)
            RunDecoder(GetEmbedding(inputIds[..^1]));
        var logits = RunDecoder(GetEmbedding(inputIds[^1..]));
        _generatedTokens.Clear();
        _generatedTokenSet.Clear();

        var stopwatch = Stopwatch.StartNew();
        var generatedText = new System.Text.StringBuilder();
        var loopDetected = false;
        var thinkingBudgetExceeded = false;
        var thinkClosed = false;
        var wasCancelled = false;
        var stopReason = "达到最大生成长度";
        long generationLimit = config.MaxNewTokens +
            (config.ThinkMode && config.ThinkingBudget > 0 && _thinkEndTokenId.HasValue
                ? config.ThinkingBudget : 0);

        for (long step = 0; step < generationLimit; step++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                wasCancelled = true;
                stopReason = "用户停止生成";
                break;
            }

            int tokenId;
            if (config.ThinkMode && config.ThinkingBudget > 0 && !thinkClosed && _thinkEndTokenId.HasValue &&
                step >= config.ThinkingBudget)
            {
                tokenId = _thinkEndTokenId.Value;
                thinkingBudgetExceeded = true;
            }
            else
            {
                if (config.DoSample)
                {
                    ApplyRepetitionPenalty(logits, _generatedTokenSet, config.RepetitionPenalty);
                    tokenId = SampleToken(logits, config);
                }
                else
                {
                    tokenId = ArgMax(logits);
                }
            }

            if (_thinkEndTokenId == tokenId)
            {
                thinkClosed = true;
                if (config.ThinkMode && config.ThinkingBudget > 0)
                    generationLimit = step + config.MaxNewTokens + 1L;
            }
            if (tokenId is ImEndTokenId or EndOfTextTokenId)
            {
                stopReason = $"遇到 EOS token ({tokenId})";
                break;
            }

            _generatedTokens.Add(tokenId);
            _generatedTokenSet.Add(tokenId);
            var tokenText = _tokenizer.Decode([checked((uint)tokenId)]);
            generatedText.Append(tokenText);
            if (config.Stream) onToken?.Invoke(tokenText);

            if (config.EnableLoopDetection &&
                DetectLoop(_generatedTokens, config.LoopCheckWindow, config.LoopRepeatThreshold))
            {
                loopDetected = true;
                stopReason = "检测到重复输出循环";
                _logger?.LogWarning("Generation stopped after detecting a repeated token loop");
                break;
            }

            if (_historyLength >= _config.MaxSequenceLength)
            {
                stopReason = "达到最大上下文长度";
                break;
            }
            logits = RunDecoder(GetEmbedding([tokenId]));
        }

        wasCancelled |= cancellationToken.IsCancellationRequested;
        if (wasCancelled)
        {
            stopReason = "用户停止生成";
        }
        stopwatch.Stop();
        var totalTokens = _generatedTokens.Count;
        var totalSeconds = stopwatch.Elapsed.TotalSeconds;
        return (generatedText.ToString(), new GenerationStats(
            totalTokens, totalSeconds,
            totalSeconds > 0 ? totalTokens / totalSeconds : 0,
            totalTokens > 0 ? totalSeconds / totalTokens : 0,
            loopDetected, thinkingBudgetExceeded, wasCancelled, stopReason));
    }

    public void Dispose()
    {
        _previousDecoderOutputs?.Dispose();
        _previousDecoderOutputs = null;

        if (_emptyKeys is not null)
            foreach (var v in _emptyKeys) v?.Dispose();
        if (_emptyValues is not null)
            foreach (var v in _emptyValues) v?.Dispose();

        _embeddingSession?.Dispose();
        _decoderSession?.Dispose();
    }

    // ==================== 热路径 ====================

    private DenseTensor<float> GetEmbedding(long[] inputIds)
    {
        using var inputOrt = OrtValue.CreateTensorValueFromMemory<long>(
            _memoryInfo, new Memory<long>(inputIds), new long[] { 1, inputIds.Length });

        var inputs = new Dictionary<string, OrtValue> { { _embeddingInputName, inputOrt } };
        using var outputs = _embeddingSession.Run(
            new RunOptions(), inputs, new[] { _embeddingOutputName });

        var span = outputs[0].GetTensorDataAsSpan<float>();
        var shapeLong = outputs[0].GetTensorTypeAndShape().Shape;
        var shapeInt = new int[shapeLong.Length];
        for (var i = 0; i < shapeLong.Length; i++) shapeInt[i] = (int)shapeLong[i];

        var data = new float[span.Length];
        span.CopyTo(data);
        return new DenseTensor<float>(new Memory<float>(data), shapeInt);
    }

    private float[] RunDecoder(DenseTensor<float> hiddenStates)
    {
        _historyLenBuffer[0] = _historyLength;
        _idsLenBuffer[0] = hiddenStates.Dimensions[1];
        _attnBuffer[0] = _attentionMask;

        using var hiddenOrt = OrtValue.CreateTensorValueFromMemory<float>(
            _memoryInfo, hiddenStates.Buffer, ToLongArray(hiddenStates.Dimensions));
        using var historyLenOrt = OrtValue.CreateTensorValueFromMemory<long>(
            _memoryInfo, new Memory<long>(_historyLenBuffer), new long[] { 1 });
        using var idsLenOrt = OrtValue.CreateTensorValueFromMemory<long>(
            _memoryInfo, new Memory<long>(_idsLenBuffer), new long[] { 1 });
        using var attnOrt = OrtValue.CreateTensorValueFromMemory<sbyte>(
            _memoryInfo, new Memory<sbyte>(_attnBuffer), new long[] { 1 });

        var inputs = new Dictionary<string, OrtValue>(2 * _numLayers + 4)
        {
            { "hidden_states", hiddenOrt },
            { "history_len", historyLenOrt },
            { "ids_len", idsLenOrt },
            { "attention_mask", attnOrt }
        };

        if (_previousDecoderOutputs is not null)
        {
            for (var layer = 0; layer < _numLayers; layer++)
            {
                inputs[$"in_key_{layer}"] = _previousDecoderOutputs[layer];
                inputs[$"in_value_{layer}"] = _previousDecoderOutputs[_numLayers + layer];
            }
        }
        else
        {
            for (var layer = 0; layer < _numLayers; layer++)
            {
                inputs[$"in_key_{layer}"] = _emptyKeys[layer];
                inputs[$"in_value_{layer}"] = _emptyValues[layer];
            }
        }

        var outputs = _decoderSession.Run(new RunOptions(), inputs, _decoderOutputNames);

        var oldOutputs = _previousDecoderOutputs;
        _previousDecoderOutputs = outputs;
        oldOutputs?.Dispose();

        var logitsSpan = outputs[^2].GetTensorDataAsSpan<float>();
        var logits = logitsSpan.ToArray();

        _historyLength = outputs[^1].GetTensorDataAsSpan<long>()[0];
        _attentionMask = 0;

        return logits;
    }

    private static long[] ToLongArray(ReadOnlySpan<int> values)
    {
        var result = new long[values.Length];
        for (var i = 0; i < values.Length; i++) result[i] = values[i];
        return result;
    }

    // ==================== 原有逻辑，保持不变 ====================

    private void ValidateDecoderContract()
    {
        var expectedInputCount = 2 * _numLayers + 4;
        if (_decoderInputNames.Count != expectedInputCount)
            throw new InvalidDataException(
                $"Decoder model must have {expectedInputCount} inputs for {_numLayers} layers; found {_decoderInputNames.Count}.");

        var requiredNames = new HashSet<string>(StringComparer.Ordinal)
        { "hidden_states", "history_len", "ids_len", "attention_mask" };
        for (var layer = 0; layer < _numLayers; layer++)
        {
            requiredNames.Add($"in_key_{layer}");
            requiredNames.Add($"in_value_{layer}");
        }
        if (!requiredNames.SetEquals(_decoderInputNames))
        {
            var missing = requiredNames.Except(_decoderInputNames);
            throw new InvalidDataException($"Decoder input names do not match. Missing: {string.Join(", ", missing)}.");
        }
        if (_decoderOutputNames.Length != 2 * _numLayers + 2)
            throw new InvalidDataException(
                $"Decoder model must return exactly {2 * _numLayers + 2} outputs for {_numLayers} layers; found {_decoderOutputNames.Length}.");
    }

    private static void ValidateFiles(AssistantConfig config)
    {
        foreach (var path in new[] { config.ModelConfigPath, config.TokenizerPath, config.EmbeddingModelPath, config.DecoderModelPath })
            if (!File.Exists(path)) throw new FileNotFoundException($"Required model file was not found: {path}", path);
        if (config.MaxSequenceLength <= 0 || config.IntraOpNumThreads <= 0 || config.InterOpNumThreads <= 0)
            throw new ArgumentOutOfRangeException(nameof(config), "Sequence length and thread counts must be positive.");
    }

    private static int ReadRequiredInt(JsonElement json, string name) =>
        json.TryGetProperty(name, out var value) && value.TryGetInt32(out var result)
            ? result : throw new InvalidDataException($"config.json must contain integer property '{name}'.");

    private static int ReadOptionalInt(JsonElement json, string name, int fallback) =>
        json.TryGetProperty(name, out var value) && value.TryGetInt32(out var result) ? result : fallback;

    private long[] EncodeInput(string text) => _tokenizer.Encode(text).Select(id => (long)id).ToArray();

    private static void ValidateGenerationConfig(GenerationConfig config)
    {
        if (config.MaxNewTokens <= 0 || config.LoopCheckWindow <= 0 || config.LoopRepeatThreshold <= 0 ||
            config.ThinkingBudget < 0 || config.MaxTurns < 0 || config.MaxHistoryChars < 0 ||
            config.MaxHistoryTokens < 0 || config.TopK < 0 || config.Temperature < 0 ||
            config.TopP <= 0 || config.TopP > 1 || config.RepetitionPenalty <= 0 ||
            config.RepetitionThreshold is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(config), "Generation configuration contains an invalid value.");
    }

    private static int ArgMax(IReadOnlyList<float> logits)
    {
        var maxIndex = 0; var maxValue = logits[0];
        for (var i = 1; i < logits.Count; i++)
            if (logits[i] > maxValue) { maxValue = logits[i]; maxIndex = i; }
        return maxIndex;
    }

    private static void ApplyRepetitionPenalty(float[] logits, IReadOnlyCollection<int> generatedTokens, double penalty)
    {
        if (penalty == 1) return;
        foreach (var tokenId in generatedTokens)
        {
            if ((uint)tokenId >= (uint)logits.Length) continue;
            logits[tokenId] = logits[tokenId] > 0
                ? (float)(logits[tokenId] / penalty)
                : (float)(logits[tokenId] * penalty);
        }
    }

    private static int SampleToken(float[] logits, GenerationConfig config)
    {
        var temperature = config.Temperature > 0 ? config.Temperature : 1.0;
        var invTemperature = 1.0 / temperature;
        if (config.TopK == 0 && config.TopP == 1) return SamplePureTemperature(logits, invTemperature);
        var k = config.TopK > 0 ? Math.Min(config.TopK, logits.Length) : logits.Length;
        return k <= 256 ? SampleSmallK(logits, config, invTemperature, k)
                        : SampleLargeK(logits, config, invTemperature, k);
    }

    private static int SamplePureTemperature(float[] logits, double invTemperature)
    {
        var max = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        var scaledMax = max * invTemperature;
        var total = 0.0;
        for (var i = 0; i < logits.Length; i++) total += Math.Exp(logits[i] * invTemperature - scaledMax);
        if (total <= 0 || double.IsNaN(total) || double.IsInfinity(total)) return ArgMax(logits);
        var sample = Random.Shared.NextDouble() * total;
        for (var i = 0; i < logits.Length; i++)
        {
            sample -= Math.Exp(logits[i] * invTemperature - scaledMax);
            if (sample <= 0) return i;
        }
        return logits.Length - 1;
    }

    private static int SampleSmallK(float[] logits, GenerationConfig config, double invTemperature, int k)
    {
        Span<int> heapIdx = stackalloc int[k]; Span<float> heapVal = stackalloc float[k];
        var heapSize = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            var v = logits[i];
            if (heapSize < k)
            {
                heapIdx[heapSize] = i; heapVal[heapSize] = v;
                SiftUp(heapIdx, heapVal, heapSize); heapSize++;
            }
            else if (v > heapVal[0])
            {
                heapIdx[0] = i; heapVal[0] = v;
                SiftDown(heapIdx, heapVal, heapSize, 0);
            }
        }
        var count = heapSize;
        Span<int> orderedIdx = stackalloc int[count]; Span<double> orderedScaled = stackalloc double[count];
        Span<double> weights = stackalloc double[count];
        for (var i = count - 1; i >= 0; i--)
        {
            orderedIdx[i] = heapIdx[0]; orderedScaled[i] = heapVal[0] * invTemperature;
            heapSize--;
            if (heapSize > 0)
            {
                heapIdx[0] = heapIdx[heapSize]; heapVal[0] = heapVal[heapSize];
                SiftDown(heapIdx, heapVal, heapSize, 0);
            }
        }
        return FinishSampling(orderedIdx, orderedScaled, weights, config, logits);
    }

    private static int SampleLargeK(float[] logits, GenerationConfig config, double invTemperature, int k)
    {
        var pq = new PriorityQueue<int, float>(k);
        for (var i = 0; i < logits.Length; i++)
            if (pq.Count < k) pq.Enqueue(i, logits[i]);
            else if (logits[i] > pq.Peek()) pq.EnqueueDequeue(i, logits[i]);
        var count = pq.Count;
        Span<int> idx = stackalloc int[count]; Span<double> scaled = stackalloc double[count];
        Span<double> weights = stackalloc double[count];
        for (var i = count - 1; i >= 0; i--)
        {
            pq.TryDequeue(out var ix, out var v);
            idx[i] = ix; scaled[i] = v * invTemperature;
        }
        return FinishSampling(idx, scaled, weights, config, logits);
    }

    private static int FinishSampling(
        ReadOnlySpan<int> indices, ReadOnlySpan<double> scaledLogits,
        Span<double> weights, GenerationConfig config, float[] originalLogits)
    {
        var maxScaled = scaledLogits[0]; var total = 0.0;
        for (var i = 0; i < scaledLogits.Length; i++)
        {
            var w = Math.Exp(scaledLogits[i] - maxScaled); weights[i] = w; total += w;
        }
        if (total <= 0 || double.IsNaN(total) || double.IsInfinity(total)) return ArgMax(originalLogits);
        if (config.TopP < 1)
        {
            var cumulative = 0.0; var keep = 0;
            for (; keep < weights.Length; keep++)
            {
                cumulative += weights[keep];
                if (cumulative / total >= config.TopP) { keep++; break; }
            }
            if (keep < weights.Length)
            {
                total = cumulative;
                for (var i = keep; i < weights.Length; i++) weights[i] = 0;
            }
        }
        var sample = Random.Shared.NextDouble() * total;
        for (var i = 0; i < weights.Length; i++)
        { sample -= weights[i]; if (sample <= 0) return indices[i]; }
        return indices[0];
    }

    private static void SiftUp(Span<int> hi, Span<float> hv, int pos)
    {
        while (pos > 0)
        {
            var p = (pos - 1) >> 1;
            if (hv[p] <= hv[pos]) break;
            (hi[p], hi[pos]) = (hi[pos], hi[p]); (hv[p], hv[pos]) = (hv[pos], hv[p]);
            pos = p;
        }
    }

    private static void SiftDown(Span<int> hi, Span<float> hv, int size, int pos)
    {
        while (true)
        {
            var l = (pos << 1) + 1; if (l >= size) break;
            var r = l + 1; var s = r < size && hv[r] < hv[l] ? r : l;
            if (hv[pos] <= hv[s]) break;
            (hi[pos], hi[s]) = (hi[s], hi[pos]); (hv[pos], hv[s]) = (hv[s], hv[pos]);
            pos = s;
        }
    }

    private static bool DetectLoop(IReadOnlyList<int> tokens, int window, int repeatThreshold)
    {
        if (tokens.Count < window) return false;
        var start = tokens.Count - window;
        for (var pl = 3; pl <= 9; pl++)
        {
            if (window < pl * repeatThreshold) continue;
            var count = 1; var cursor = tokens.Count - pl * 2; var lastStart = tokens.Count - pl;
            while (cursor >= start && count < repeatThreshold)
            {
                var match = true;
                for (var o = 0; o < pl; o++)
                    if (tokens[cursor + o] != tokens[lastStart + o]) { match = false; break; }
                if (!match) break;
                count++; cursor -= pl;
            }
            if (count >= repeatThreshold) return true;
        }
        return false;
    }
}