using System.Diagnostics;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using Tokenizers.DotNet;

namespace Qwen3.Inference;

public sealed record GenerationStats(
    int TotalTokens,
    double TotalTimeSeconds,
    double TokensPerSecond,
    double AverageTimePerTokenSeconds,
    bool LoopDetected,
    bool ThinkingBudgetExceeded,
    bool WasCancelled = false);

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
    private readonly int _numLayers;
    private readonly int _numKeyValueHeads;
    private readonly int _headDim;
    private readonly string[] _decoderOutputNames;
    private readonly HashSet<string> _decoderInputNames;
    private DenseTensor<float>[] _keys = [];
    private DenseTensor<float>[] _values = [];
    private List<int> _generatedTokens = [];
    private long _historyLength;
    private sbyte _attentionMask = 1;
    private readonly int? _thinkEndTokenId;

    public long HistoryLength => _historyLength;
    public string EngineId => "qwen3-onnx";

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
        {
            throw new InvalidDataException("config.json contains invalid attention or hidden dimensions.");
        }
        _numKeyValueHeads = ReadOptionalInt(modelConfig, "num_key_value_heads", attentionHeads);
        _headDim = ReadOptionalInt(modelConfig, "head_dim", hiddenSize / attentionHeads);
        if (attentionHeads <= 0 || _numLayers <= 0 || _numKeyValueHeads <= 0 || _headDim <= 0)
        {
            throw new InvalidDataException("config.json contains invalid model dimensions.");
        }

        _tokenizer = new Tokenizer(config.TokenizerPath);
        var thinkEndTokens = _tokenizer.Encode("</think>").ToArray();
        _thinkEndTokenId = thinkEndTokens.Length == 0 ? null : checked((int)thinkEndTokens[^1]);
        _logger?.LogInformation("Loading Qwen3 ONNX model from {ModelPath}", config.ModelPath);

        using var options = new SessionOptions
        {
            IntraOpNumThreads = config.IntraOpNumThreads,
            InterOpNumThreads = config.InterOpNumThreads,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        options.AddSessionConfigEntry("session.enable_quant_qdq_cleanup", "1");
        options.AddSessionConfigEntry("session.qdq_matmulnbits_accuracy_level", "4");
        options.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
        options.AddSessionConfigEntry("session.inter_op.allow_spinning", "1");
        options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");

        InferenceSession? embeddingSession = null;
        InferenceSession? decoderSession = null;
        try
        {
            embeddingSession = new InferenceSession(config.EmbeddingModelPath, options);
            decoderSession = new InferenceSession(config.DecoderModelPath, options);
        }
        catch
        {
            embeddingSession?.Dispose();
            decoderSession?.Dispose();
            throw;
        }
        _embeddingSession = embeddingSession;
        _decoderSession = decoderSession;

        var embeddingInputs = _embeddingSession.InputMetadata.Keys.ToArray();
        if (embeddingInputs.Length != 1)
        {
            Dispose();
            throw new InvalidDataException($"Embedding model must have exactly one input; found {embeddingInputs.Length}.");
        }
        _embeddingInputName = embeddingInputs[0];
        _decoderInputNames = _decoderSession.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal);
        _decoderOutputNames = _decoderSession.OutputMetadata.Keys.ToArray();
        try
        {
            ValidateDecoderContract();
            ResetState();
        }
        catch
        {
            Dispose();
            throw;
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
            {
                prompt.Append("<|im_start|>assistant\n").Append(turn.Assistant).Append("<|im_end|>\n");
            }
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
        _keys = Enumerable.Range(0, _numLayers)
            .Select(_ => new DenseTensor<float>(Array.Empty<float>(), [_numKeyValueHeads, 1, _headDim, 0]))
            .ToArray();
        _values = Enumerable.Range(0, _numLayers)
            .Select(_ => new DenseTensor<float>(Array.Empty<float>(), [_numKeyValueHeads, 1, 0, _headDim]))
            .ToArray();
        _historyLength = 0;
        _attentionMask = 1;
        _generatedTokens.Clear();
    }

    public void Prefill(string text)
    {
        var ids = EncodeInput(text);
        if (ids.Length == 0)
        {
            return;
        }

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
        {
            throw new InvalidOperationException("The prompt produced no tokens.");
        }
        if (_historyLength + inputIds.Length > _config.MaxSequenceLength)
        {
            throw new InvalidOperationException(
                $"Prompt and KV cache exceed the configured maximum sequence length ({_config.MaxSequenceLength}).");
        }

        if (inputIds.Length > 1)
        {
            RunDecoder(GetEmbedding(inputIds[..^1]));
        }
        var logits = RunDecoder(GetEmbedding(inputIds[^1..]));
        _generatedTokens.Clear();

        var stopwatch = Stopwatch.StartNew();
        var generatedText = new System.Text.StringBuilder();
        var loopDetected = false;
        var thinkingBudgetExceeded = false;
        var thinkClosed = false;
        var wasCancelled = false;
        long generationLimit = config.MaxNewTokens +
            (config.ThinkMode && config.ThinkingBudget > 0 && _thinkEndTokenId.HasValue
                ? config.ThinkingBudget
                : 0);

        for (long step = 0; step < generationLimit; step++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                wasCancelled = true;
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
                    ApplyRepetitionPenalty(logits, _generatedTokens, config.RepetitionPenalty);
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
                {
                    generationLimit = step + config.MaxNewTokens + 1L;
                }
            }
            if (tokenId is ImEndTokenId or EndOfTextTokenId)
            {
                break;
            }

            _generatedTokens.Add(tokenId);
            var tokenText = _tokenizer.Decode([checked((uint)tokenId)]);
            generatedText.Append(tokenText);
            if (config.Stream)
            {
                onToken?.Invoke(tokenText);
            }

            if (config.EnableLoopDetection &&
                DetectLoop(_generatedTokens, config.LoopCheckWindow, config.LoopRepeatThreshold))
            {
                loopDetected = true;
                _logger?.LogWarning("Generation stopped after detecting a repeated token loop");
                break;
            }

            if (_historyLength >= _config.MaxSequenceLength)
            {
                break;
            }
            logits = RunDecoder(GetEmbedding([tokenId]));
        }

        wasCancelled |= cancellationToken.IsCancellationRequested;
        stopwatch.Stop();
        var totalTokens = _generatedTokens.Count;
        var totalSeconds = stopwatch.Elapsed.TotalSeconds;
        return (generatedText.ToString(), new GenerationStats(
            totalTokens,
            totalSeconds,
            totalSeconds > 0 ? totalTokens / totalSeconds : 0,
            totalTokens > 0 ? totalSeconds / totalTokens : 0,
            loopDetected,
            thinkingBudgetExceeded,
            wasCancelled));
    }

    public void Dispose()
    {
        _embeddingSession?.Dispose();
        _decoderSession?.Dispose();
    }

    private static void ValidateFiles(AssistantConfig config)
    {
        foreach (var path in new[] { config.ModelConfigPath, config.TokenizerPath, config.EmbeddingModelPath, config.DecoderModelPath })
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Required model file was not found: {path}", path);
            }
        }
        if (config.MaxSequenceLength <= 0 || config.IntraOpNumThreads <= 0 || config.InterOpNumThreads <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(config), "Sequence length and thread counts must be positive.");
        }
    }

    private void ValidateDecoderContract()
    {
        var expectedInputCount = 2 * _numLayers + 4;
        if (_decoderInputNames.Count != expectedInputCount)
        {
            throw new InvalidDataException(
                $"Decoder model must have {expectedInputCount} inputs for {_numLayers} layers; found {_decoderInputNames.Count}.");
        }

        var requiredNames = new HashSet<string>(StringComparer.Ordinal)
        {
            "hidden_states", "history_len", "ids_len", "attention_mask"
        };
        for (var layer = 0; layer < _numLayers; layer++)
        {
            requiredNames.Add($"in_key_{layer}");
            requiredNames.Add($"in_value_{layer}");
        }

        if (!requiredNames.SetEquals(_decoderInputNames))
        {
            var missing = requiredNames.Except(_decoderInputNames);
            throw new InvalidDataException($"Decoder input names do not match the expected contract. Missing: {string.Join(", ", missing)}.");
        }
        if (_decoderOutputNames.Length != 2 * _numLayers + 2)
        {
            throw new InvalidDataException(
                $"Decoder model must return exactly {2 * _numLayers + 2} outputs for {_numLayers} layers; found {_decoderOutputNames.Length}.");
        }
    }

    private static int ReadRequiredInt(JsonElement json, string name) =>
        json.TryGetProperty(name, out var value) && value.TryGetInt32(out var result)
            ? result
            : throw new InvalidDataException($"config.json must contain integer property '{name}'.");

    private static int ReadOptionalInt(JsonElement json, string name, int fallback) =>
        json.TryGetProperty(name, out var value) && value.TryGetInt32(out var result) ? result : fallback;

    private long[] EncodeInput(string text) => _tokenizer.Encode(text).Select(id => (long)id).ToArray();

    private DenseTensor<float> GetEmbedding(long[] inputIds)
    {
        var input = NamedOnnxValue.CreateFromTensor(
            _embeddingInputName, new DenseTensor<long>(inputIds, new int[] { 1, inputIds.Length }));
        using var outputs = _embeddingSession.Run([input]);
        return CopyTensor(outputs[0].AsTensor<float>());
    }

    private float[] RunDecoder(DenseTensor<float> hiddenStates)
    {
        var inputs = new List<NamedOnnxValue>(_decoderInputNames.Count)
        {
            NamedOnnxValue.CreateFromTensor("hidden_states", hiddenStates)
        };
        foreach (var name in _decoderInputNames)
        {
            if (name.StartsWith("in_key_", StringComparison.Ordinal) &&
                int.TryParse(name.AsSpan("in_key_".Length), out var keyLayer))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, _keys[keyLayer]));
            }
            else if (name.StartsWith("in_value_", StringComparison.Ordinal) &&
                     int.TryParse(name.AsSpan("in_value_".Length), out var valueLayer))
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(name, _values[valueLayer]));
            }
        }
        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "history_len", new DenseTensor<long>(new long[] { _historyLength }, new int[] { 1 })));
        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "ids_len", new DenseTensor<long>(new long[] { hiddenStates.Dimensions[1] }, new int[] { 1 })));
        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "attention_mask", new DenseTensor<sbyte>(new sbyte[] { _attentionMask }, new int[] { 1 })));

        using var outputs = _decoderSession.Run(inputs);
        if (outputs.Count != _decoderOutputNames.Length)
        {
            throw new InvalidDataException($"Decoder returned {outputs.Count} outputs; expected {_decoderOutputNames.Length}.");
        }

        for (var layer = 0; layer < _numLayers; layer++)
        {
            _keys[layer] = CopyTensor(outputs[layer].AsTensor<float>());
            _values[layer] = CopyTensor(outputs[_numLayers + layer].AsTensor<float>());
        }

        var logitsTensor = outputs[^2].AsTensor<float>();
        var logitsDimensions = logitsTensor.Dimensions.ToArray();
        if (logitsDimensions.Length < 2 || logitsDimensions[^1] <= 0)
        {
            throw new InvalidDataException("Decoder logits must contain a vocabulary dimension.");
        }
        var logitsFlat = logitsTensor.ToArray();
        var vocabSize = logitsDimensions[^1];
        var sequenceLength = logitsDimensions.Length >= 3 ? logitsDimensions[^2] : 1;
        var logits = new float[vocabSize];
        Array.Copy(logitsFlat, (sequenceLength - 1) * vocabSize, logits, 0, vocabSize);

        var lengthTensor = outputs[^1];
        _historyLength = lengthTensor.ElementType switch
        {
            TensorElementType.Int64 => lengthTensor.AsTensor<long>().GetValue(0),
            TensorElementType.Int32 => lengthTensor.AsTensor<int>().GetValue(0),
            _ => throw new InvalidDataException($"Unsupported KV sequence length tensor type: {lengthTensor.ElementType}.")
        };
        _attentionMask = 0;
        return logits;
    }

    private static DenseTensor<float> CopyTensor(Tensor<float> tensor) =>
        new(tensor.ToArray(), tensor.Dimensions.ToArray());

    private static void ValidateGenerationConfig(GenerationConfig config)
    {
        if (config.MaxNewTokens <= 0 || config.LoopCheckWindow <= 0 || config.LoopRepeatThreshold <= 0 ||
            config.ThinkingBudget < 0 || config.MaxTurns < 0 || config.MaxHistoryChars < 0 ||
            config.MaxHistoryTokens < 0 || config.TopK < 0 || config.Temperature < 0 ||
            config.TopP <= 0 || config.TopP > 1 || config.RepetitionPenalty <= 0 ||
            config.RepetitionThreshold is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(config), "Generation configuration contains an invalid value.");
        }
    }

    private static int ArgMax(IReadOnlyList<float> logits)
    {
        var maxIndex = 0;
        for (var index = 1; index < logits.Count; index++)
        {
            if (logits[index] > logits[maxIndex])
            {
                maxIndex = index;
            }
        }

        return maxIndex;
    }

    private static void ApplyRepetitionPenalty(float[] logits, IEnumerable<int> generatedTokens, double penalty)
    {
        if (penalty == 1)
        {
            return;
        }

        foreach (var tokenId in generatedTokens.Distinct())
        {
            if (tokenId < 0 || tokenId >= logits.Length)
            {
                continue;
            }

            logits[tokenId] = logits[tokenId] > 0
                ? (float)(logits[tokenId] / penalty)
                : (float)(logits[tokenId] * penalty);
        }
    }

    private static int SampleToken(float[] logits, GenerationConfig config)
    {
        var temperature = config.Temperature > 0 ? config.Temperature : 1.0;
        var maxLogit = logits.Max() / temperature;
        var probabilities = logits.Select(value => Math.Exp(value / temperature - maxLogit)).ToArray();
        var total = probabilities.Sum();
        if (total <= 0 || double.IsNaN(total) || double.IsInfinity(total))
        {
            return ArgMax(logits);
        }
        for (var index = 0; index < probabilities.Length; index++)
        {
            probabilities[index] /= total;
        }

        if (config.TopK > 0 && config.TopK < probabilities.Length)
        {
            var keep = probabilities.Select((probability, index) => (probability, index))
                .OrderByDescending(item => item.probability).Take(config.TopK)
                .Select(item => item.index).ToHashSet();
            for (var index = 0; index < probabilities.Length; index++)
            {
                if (!keep.Contains(index))
                {
                    probabilities[index] = 0;
                }
            }
        }

        if (config.TopP < 1)
        {
            var sorted = probabilities.Select((probability, index) => (probability, index))
                .OrderByDescending(item => item.probability).ToArray();
            var cumulative = 0.0;
            var keep = new HashSet<int>();
            foreach (var item in sorted)
            {
                keep.Add(item.index);
                cumulative += item.probability;
                if (cumulative >= config.TopP)
                {
                    break;
                }
            }
            for (var index = 0; index < probabilities.Length; index++)
            {
                if (!keep.Contains(index))
                {
                    probabilities[index] = 0;
                }
            }
        }

        total = probabilities.Sum();
        if (total <= 0)
        {
            return ArgMax(logits);
        }
        var sample = Random.Shared.NextDouble() * total;
        for (var index = 0; index < probabilities.Length; index++)
        {
            sample -= probabilities[index];
            if (sample <= 0)
            {
                return index;
            }
        }

        return probabilities.Length - 1;
    }

    private static bool DetectLoop(IReadOnlyList<int> tokens, int window, int repeatThreshold)
    {
        if (tokens.Count < window)
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

            var count = 1;
            var cursor = tokens.Count - patternLength * 2;
            while (cursor >= start)
            {
                var matches = true;
                for (var offset = 0; offset < patternLength; offset++)
                {
                    if (tokens[cursor + offset] != tokens[tokens.Count - patternLength + offset])
                    {
                        matches = false;
                        break;
                    }
                }
                if (!matches)
                {
                    break;
                }

                count++;
                cursor -= patternLength;
            }
            if (count >= repeatThreshold)
            {
                return true;
            }
        }

        return false;
    }
}
