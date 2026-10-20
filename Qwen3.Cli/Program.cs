using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Console;
using Qwen3.Cli;
using Qwen3.Inference;
using System.CommandLine;
using System.Text;

var rootCommand = BuildCommand();
return await rootCommand.Parse(args).InvokeAsync();

static RootCommand BuildCommand()
{
    var root = new RootCommand("Run one Qwen-compatible model inference request.");
    var question = new Argument<string>("question")
    {
        Description = "Question or prompt to send to the model."
    };
    question.Validators.Add(result =>
    {
        if (result.Errors.Any())
        {
            return;
        }
        if (string.IsNullOrWhiteSpace(result.GetValueOrDefault<string>()))
        {
            result.AddError("Question must not be empty.");
        }
    });
    root.Arguments.Add(question);

    var modelPath = new Option<string?>("--model-path") { Description = "Directory containing tokenizer.json, config.json, and ONNX model files." };
    var engineId = new Option<string?>("--engine") { Description = "Inference engine identifier (default: qwen3-onnx)." };
    var scene = new Option<string?>("--scene") { Description = "Generation scene: fast_qa, deep, roleplay, precise, or chitchat." };
    scene.AcceptOnlyFromAmong("fast_qa", "deep", "roleplay", "precise", "chitchat");
    var thinkMode = new Option<string>("--think-mode")
    {
        Description = "Thinking mode: auto uses the selected scene default.",
        DefaultValueFactory = _ => "auto"
    };
    thinkMode.AcceptOnlyFromAmong("auto", "on", "off");
    var systemPrompt = new Option<string?>("--system-prompt") { Description = "Override the system prompt." };
    var logLevel = new Option<string>("--log-level")
    {
        Description = "Minimum log level: trace, debug, information, warning, error, or none.",
        DefaultValueFactory = _ => "information"
    };
    logLevel.AcceptOnlyFromAmong("trace", "debug", "information", "warning", "error", "none");
    var logFile = new Option<string?>("--log-file") { Description = "Append structured JSON logs to this file." };

    var maxNewTokens = IntegerOption("--max-new-tokens", "Maximum number of generated tokens.");
    var thinkingBudget = IntegerOption("--thinking-budget", "Maximum thinking tokens before closing the think section.");
    var maxTurns = IntegerOption("--max-turns", "Maximum number of retained conversation turns.");
    var maxHistoryChars = IntegerOption("--max-history-chars", "Maximum characters kept per assistant history entry.");
    var maxHistoryTokens = IntegerOption("--max-history-tokens", "Maximum token count retained in conversation history.");
    var topK = IntegerOption("--top-k", "Top-k sampling limit; zero disables top-k filtering.");
    var loopCheckWindow = IntegerOption("--loop-check-window", "Number of recent generated tokens checked for loops.");
    var loopRepeatThreshold = IntegerOption("--loop-repeat-threshold", "Repeated token-pattern count that stops generation.");

    var temperature = DoubleOption("--temperature", "Sampling temperature; zero leaves logits unscaled.");
    var topP = DoubleOption("--top-p", "Nucleus sampling probability in (0, 1].");
    var repetitionPenalty = DoubleOption("--repetition-penalty", "Repetition penalty; must be greater than zero.");
    var repetitionThreshold = DoubleOption("--repetition-threshold", "Answer similarity threshold in [0, 1].");

    var multiTurn = new Option<bool>("--multi-turn") { Description = "Enable multi-turn conversation memory." };
    var singleTurn = new Option<bool>("--single-turn") { Description = "Disable multi-turn conversation memory." };
    var stream = new Option<bool>("--stream") { Description = "Stream generated tokens to standard output." };
    var noStream = new Option<bool>("--no-stream") { Description = "Buffer the answer and print it after generation." };
    var doSample = new Option<bool>("--sample") { Description = "Enable sampling." };
    var noSample = new Option<bool>("--no-sample") { Description = "Disable sampling and use greedy decoding." };
    var showThinking = new Option<bool>("--show-thinking") { Description = "Show generated thinking text." };
    var hideThinking = new Option<bool>("--hide-thinking") { Description = "Hide generated thinking text." };
    var stripThinking = new Option<bool>("--strip-thinking-in-history") { Description = "Remove thinking text before storing conversation history." };
    var keepThinking = new Option<bool>("--keep-thinking-in-history") { Description = "Keep thinking text in conversation history." };
    var sanitizeHistory = new Option<bool>("--sanitize-history") { Description = "Sanitize roleplay text before storing history." };
    var noSanitizeHistory = new Option<bool>("--no-sanitize-history") { Description = "Do not sanitize stored history." };
    var resetOnFact = new Option<bool>("--reset-on-fact-question") { Description = "Clear history before standalone fact questions." };
    var noResetOnFact = new Option<bool>("--no-reset-on-fact-question") { Description = "Keep history for fact questions." };
    var repetitionCheck = new Option<bool>("--repetition-check") { Description = "Discard answers that repeat recent answers." };
    var noRepetitionCheck = new Option<bool>("--no-repetition-check") { Description = "Disable repeated-answer checking." };
    var loopDetection = new Option<bool>("--loop-detection") { Description = "Stop generation when a token loop is detected." };
    var noLoopDetection = new Option<bool>("--no-loop-detection") { Description = "Disable token-loop detection." };

    root.Options.Add(modelPath);
    root.Options.Add(engineId);
    root.Options.Add(scene);
    root.Options.Add(thinkMode);
    root.Options.Add(systemPrompt);
    root.Options.Add(logLevel);
    root.Options.Add(logFile);
    root.Options.Add(maxNewTokens);
    root.Options.Add(thinkingBudget);
    root.Options.Add(maxTurns);
    root.Options.Add(maxHistoryChars);
    root.Options.Add(maxHistoryTokens);
    root.Options.Add(topK);
    root.Options.Add(loopCheckWindow);
    root.Options.Add(loopRepeatThreshold);
    root.Options.Add(temperature);
    root.Options.Add(topP);
    root.Options.Add(repetitionPenalty);
    root.Options.Add(repetitionThreshold);
    root.Options.Add(multiTurn);
    root.Options.Add(singleTurn);
    root.Options.Add(stream);
    root.Options.Add(noStream);
    root.Options.Add(doSample);
    root.Options.Add(noSample);
    root.Options.Add(showThinking);
    root.Options.Add(hideThinking);
    root.Options.Add(stripThinking);
    root.Options.Add(keepThinking);
    root.Options.Add(sanitizeHistory);
    root.Options.Add(noSanitizeHistory);
    root.Options.Add(resetOnFact);
    root.Options.Add(noResetOnFact);
    root.Options.Add(repetitionCheck);
    root.Options.Add(noRepetitionCheck);
    root.Options.Add(loopDetection);
    root.Options.Add(noLoopDetection);

    root.Validators.Add(result =>
    {
        if (result.Errors.Any())
        {
            return;
        }
        AddMutuallyExclusiveError(result, multiTurn, singleTurn, "--multi-turn", "--single-turn");
        AddMutuallyExclusiveError(result, stream, noStream, "--stream", "--no-stream");
        AddMutuallyExclusiveError(result, doSample, noSample, "--sample", "--no-sample");
        AddMutuallyExclusiveError(result, showThinking, hideThinking, "--show-thinking", "--hide-thinking");
        AddMutuallyExclusiveError(result, stripThinking, keepThinking, "--strip-thinking-in-history", "--keep-thinking-in-history");
        AddMutuallyExclusiveError(result, sanitizeHistory, noSanitizeHistory, "--sanitize-history", "--no-sanitize-history");
        AddMutuallyExclusiveError(result, resetOnFact, noResetOnFact, "--reset-on-fact-question", "--no-reset-on-fact-question");
        AddMutuallyExclusiveError(result, repetitionCheck, noRepetitionCheck, "--repetition-check", "--no-repetition-check");
        AddMutuallyExclusiveError(result, loopDetection, noLoopDetection, "--loop-detection", "--no-loop-detection");
    });

    root.SetAction(parseResult => Execute(
        parseResult,
        question, modelPath, engineId, scene, thinkMode, systemPrompt, logLevel, logFile,
        maxNewTokens, thinkingBudget, maxTurns, maxHistoryChars, maxHistoryTokens,
        topK, loopCheckWindow, loopRepeatThreshold, temperature, topP,
        repetitionPenalty, repetitionThreshold, multiTurn, singleTurn, stream, noStream,
        doSample, noSample, showThinking, hideThinking, stripThinking, keepThinking,
        sanitizeHistory, noSanitizeHistory, resetOnFact, noResetOnFact,
        repetitionCheck, noRepetitionCheck, loopDetection, noLoopDetection));

    return root;
}

static Option<int?> IntegerOption(string name, string description) =>
    new(name) { Description = description };

static Option<double?> DoubleOption(string name, string description) =>
    new(name) { Description = description };

static void AddMutuallyExclusiveError(
    System.CommandLine.Parsing.CommandResult result,
    Option<bool> first,
    Option<bool> second,
    string firstName,
    string secondName)
{
    if (result.GetValue(first) && result.GetValue(second))
    {
        result.AddError($"Options {firstName} and {secondName} cannot be used together.");
    }
}

static int Execute(
    System.CommandLine.ParseResult parse,
    Argument<string> questionArgument,
    Option<string?> modelPathOption,
    Option<string?> engineOption,
    Option<string?> sceneOption,
    Option<string> thinkModeOption,
    Option<string?> systemPromptOption,
    Option<string> logLevelOption,
    Option<string?> logFileOption,
    Option<int?> maxNewTokensOption,
    Option<int?> thinkingBudgetOption,
    Option<int?> maxTurnsOption,
    Option<int?> maxHistoryCharsOption,
    Option<int?> maxHistoryTokensOption,
    Option<int?> topKOption,
    Option<int?> loopCheckWindowOption,
    Option<int?> loopRepeatThresholdOption,
    Option<double?> temperatureOption,
    Option<double?> topPOption,
    Option<double?> repetitionPenaltyOption,
    Option<double?> repetitionThresholdOption,
    Option<bool> multiTurnOption,
    Option<bool> singleTurnOption,
    Option<bool> streamOption,
    Option<bool> noStreamOption,
    Option<bool> sampleOption,
    Option<bool> noSampleOption,
    Option<bool> showThinkingOption,
    Option<bool> hideThinkingOption,
    Option<bool> stripThinkingOption,
    Option<bool> keepThinkingOption,
    Option<bool> sanitizeHistoryOption,
    Option<bool> noSanitizeHistoryOption,
    Option<bool> resetOnFactOption,
    Option<bool> noResetOnFactOption,
    Option<bool> repetitionCheckOption,
    Option<bool> noRepetitionCheckOption,
    Option<bool> loopDetectionOption,
    Option<bool> noLoopDetectionOption)
{
    var question = parse.GetValue(questionArgument) ?? string.Empty;
    var minimumLogLevel = Enum.Parse<LogLevel>(parse.GetValue(logLevelOption) ?? "information", ignoreCase: true);
    ILoggerFactory? loggerFactory = null;
    ILogger? logger = null;

    try
    {
        loggerFactory = LoggerFactory.Create(logging =>
        {
            logging.SetMinimumLevel(minimumLogLevel);
            logging.AddSimpleConsole(options =>
            {
                options.TimestampFormat = "yyyy-MM-dd HH:mm:ss.fff ";
                options.UseUtcTimestamp = true;
                options.IncludeScopes = false;
                options.ColorBehavior = LoggerColorBehavior.Disabled;
            });
            logging.Services.Configure<ConsoleLoggerOptions>(
                options => options.LogToStandardErrorThreshold = LogLevel.Trace);
            var logFile = parse.GetValue(logFileOption);
            if (!string.IsNullOrWhiteSpace(logFile))
            {
                logging.AddProvider(new JsonFileLoggerProvider(logFile));
            }
        });
        logger = loggerFactory.CreateLogger("Qwen3.Cli");
        var sceneName = parse.GetValue(sceneOption);
        var scene = sceneName is null
            ? GenerationConfig.DetectScene(question)
            : GenerationConfig.TryParseScene(sceneName, out var parsedScene)
                ? parsedScene
                : throw new ArgumentException($"Unknown scene '{sceneName}'.");
        var baseConfig = GenerationConfig.SceneConfigs[scene];
        var thinkMode = parse.GetValue(thinkModeOption);
        var generationConfig = baseConfig with
        {
            ThinkMode = thinkMode switch
            {
                "on" => true,
                "off" => false,
                _ => baseConfig.ThinkMode
            },
            SystemPrompt = parse.GetValue(systemPromptOption) ?? baseConfig.SystemPrompt,
            MaxNewTokens = parse.GetValue(maxNewTokensOption) ?? baseConfig.MaxNewTokens,
            ThinkingBudget = parse.GetValue(thinkingBudgetOption) ?? baseConfig.ThinkingBudget,
            MaxTurns = parse.GetValue(maxTurnsOption) ?? baseConfig.MaxTurns,
            MaxHistoryChars = parse.GetValue(maxHistoryCharsOption) ?? baseConfig.MaxHistoryChars,
            MaxHistoryTokens = parse.GetValue(maxHistoryTokensOption) ?? baseConfig.MaxHistoryTokens,
            TopK = parse.GetValue(topKOption) ?? baseConfig.TopK,
            LoopCheckWindow = parse.GetValue(loopCheckWindowOption) ?? baseConfig.LoopCheckWindow,
            LoopRepeatThreshold = parse.GetValue(loopRepeatThresholdOption) ?? baseConfig.LoopRepeatThreshold,
            Temperature = parse.GetValue(temperatureOption) ?? baseConfig.Temperature,
            TopP = parse.GetValue(topPOption) ?? baseConfig.TopP,
            RepetitionPenalty = parse.GetValue(repetitionPenaltyOption) ?? baseConfig.RepetitionPenalty,
            RepetitionThreshold = parse.GetValue(repetitionThresholdOption) ?? baseConfig.RepetitionThreshold,
            MultiTurn = ResolveToggle(parse, multiTurnOption, singleTurnOption, baseConfig.MultiTurn),
            Stream = ResolveToggle(parse, streamOption, noStreamOption, baseConfig.Stream),
            DoSample = ResolveToggle(parse, sampleOption, noSampleOption, baseConfig.DoSample),
            ShowThinking = ResolveToggle(parse, showThinkingOption, hideThinkingOption, baseConfig.ShowThinking),
            StripThinkingInHistory = ResolveToggle(parse, stripThinkingOption, keepThinkingOption, baseConfig.StripThinkingInHistory),
            SanitizeHistory = ResolveToggle(parse, sanitizeHistoryOption, noSanitizeHistoryOption, baseConfig.SanitizeHistory),
            ResetOnFactQuestion = ResolveToggle(parse, resetOnFactOption, noResetOnFactOption, baseConfig.ResetOnFactQuestion),
            EnableRepetitionCheck = ResolveToggle(parse, repetitionCheckOption, noRepetitionCheckOption, baseConfig.EnableRepetitionCheck),
            EnableLoopDetection = ResolveToggle(parse, loopDetectionOption, noLoopDetectionOption, baseConfig.EnableLoopDetection)
        };
        ValidateGenerationConfig(generationConfig);

        var assistantConfig = new AssistantConfig
        {
            EngineId = parse.GetValue(engineOption) ?? "qwen3-onnx",
            ModelPath = Path.GetFullPath(parse.GetValue(modelPathOption) ??
                Environment.GetEnvironmentVariable("QWEN_MODEL_PATH") ??
                Path.Combine(AppContext.BaseDirectory, "models", "Qwen3-1.7b-int8-onnx")),
            SystemPrompt = generationConfig.SystemPrompt ?? "You are a helpful assistant."
        };

        logger.LogInformation(
            "Starting inference. Engine={EngineId}, Scene={Scene}, ModelPath={ModelPath}",
            assistantConfig.EngineId, GenerationConfig.GetSceneName(scene), assistantConfig.ModelPath);
        var thinkingFilter = generationConfig.ShowThinking
            ? null
            : new ThinkingOutputFilter(Console.Out);
        using var assistant = new ChatAssistant(
            assistantConfig,
            new InferenceEngineFactory(loggerFactory),
            loggerFactory.CreateLogger<ChatAssistant>());
        var result = assistant.Ask(
            question,
            scene,
            generationConfig: generationConfig,
            onToken: token =>
            {
                if (thinkingFilter is null)
                {
                    Console.Out.Write(token);
                }
                else
                {
                    thinkingFilter.Write(token);
                }
            });
        if (generationConfig.Stream)
        {
            thinkingFilter?.Complete();
            Console.Out.WriteLine();
        }
        else
        {
            if (generationConfig.ShowThinking)
            {
                Console.Out.WriteLine(result.Answer);
            }
            else
            {
                using var filteredAnswer = new StringWriter();
                var answerFilter = new ThinkingOutputFilter(filteredAnswer);
                answerFilter.Write(result.Answer);
                answerFilter.Complete();
                Console.Out.WriteLine(filteredAnswer.ToString());
            }
        }

        logger.LogInformation(
            "Inference completed. Tokens={TokenCount}, DurationSeconds={DurationSeconds:F3}, TokensPerSecond={TokensPerSecond:F2}",
            result.Generation.TotalTokens, result.Generation.TotalTimeSeconds, result.Generation.TokensPerSecond);
        return 0;
    }
    catch (Exception exception) when (exception is ArgumentException or InvalidOperationException)
    {
        logger?.LogError(exception, "Invalid inference request");
        Console.Error.WriteLine($"参数错误：{exception.Message}");
        return 2;
    }
    catch (FileNotFoundException exception)
    {
        logger?.LogError("Required model file was not found: {FilePath}", exception.FileName);
        Console.Error.WriteLine($"模型文件不存在：{exception.FileName}");
        return 1;
    }
    catch (NotSupportedException exception)
    {
        logger?.LogError("Inference engine selection failed: {Error}", exception.Message);
        Console.Error.WriteLine($"引擎错误：{exception.Message}");
        return 2;
    }
    catch (Exception exception)
    {
        logger?.LogError(exception, "Inference failed");
        Console.Error.WriteLine($"推理失败：{exception.Message}");
        return 1;
    }
    finally
    {
        loggerFactory?.Dispose();
    }
}

static bool ResolveToggle(
    System.CommandLine.ParseResult parse,
    Option<bool> enableOption,
    Option<bool> disableOption,
    bool defaultValue) =>
    parse.GetValue(enableOption) ? true :
    parse.GetValue(disableOption) ? false : defaultValue;

static void ValidateGenerationConfig(GenerationConfig config)
{
    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(config.MaxNewTokens, "--max-new-tokens");
    ArgumentOutOfRangeException.ThrowIfNegative(config.ThinkingBudget, "--thinking-budget");
    ArgumentOutOfRangeException.ThrowIfNegative(config.MaxTurns, "--max-turns");
    ArgumentOutOfRangeException.ThrowIfNegative(config.MaxHistoryChars, "--max-history-chars");
    ArgumentOutOfRangeException.ThrowIfNegative(config.MaxHistoryTokens, "--max-history-tokens");
    ArgumentOutOfRangeException.ThrowIfNegative(config.TopK, "--top-k");
    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(config.LoopCheckWindow, "--loop-check-window");
    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(config.LoopRepeatThreshold, "--loop-repeat-threshold");
    if (!double.IsFinite(config.Temperature) || config.Temperature < 0)
    {
        throw new ArgumentOutOfRangeException("--temperature", "Temperature must be finite and non-negative.");
    }
    if (!double.IsFinite(config.TopP) || config.TopP is <= 0 or > 1)
    {
        throw new ArgumentOutOfRangeException("--top-p", "Top-p must be greater than 0 and at most 1.");
    }
    if (!double.IsFinite(config.RepetitionPenalty) || config.RepetitionPenalty <= 0)
    {
        throw new ArgumentOutOfRangeException("--repetition-penalty", "Repetition penalty must be finite and greater than 0.");
    }
    if (!double.IsFinite(config.RepetitionThreshold) || config.RepetitionThreshold is < 0 or > 1)
    {
        throw new ArgumentOutOfRangeException("--repetition-threshold", "Repetition threshold must be between 0 and 1.");
    }
}

internal sealed class ThinkingOutputFilter(TextWriter writer)
{
    private const string StartTag = "<think>";
    private const string EndTag = "</think>";
    private readonly StringBuilder _pending = new();
    private bool _insideThinking;
    private bool _completed;

    public void Write(string text)
    {
        ObjectDisposedException.ThrowIf(_completed, this);
        _pending.Append(text);
        Process(final: false);
    }

    public void Complete()
    {
        if (_completed)
        {
            return;
        }
        Process(final: true);
        _completed = true;
    }

    private void Process(bool final)
    {
        var text = _pending.ToString();
        _pending.Clear();
        while (text.Length > 0)
        {
            var marker = _insideThinking ? EndTag : StartTag;
            var markerIndex = text.IndexOf(marker, StringComparison.Ordinal);
            if (markerIndex >= 0)
            {
                if (!_insideThinking)
                {
                    writer.Write(text[..markerIndex]);
                }
                text = text[(markerIndex + marker.Length)..];
                _insideThinking = !_insideThinking;
                continue;
            }

            if (final)
            {
                if (!_insideThinking)
                {
                    writer.Write(text);
                }
                return;
            }

            var retainedLength = LongestSuffixThatStartsMarker(text, marker);
            var visibleLength = text.Length - retainedLength;
            if (!_insideThinking && visibleLength > 0)
            {
                writer.Write(text[..visibleLength]);
            }
            if (retainedLength > 0)
            {
                _pending.Append(text.AsSpan(visibleLength));
            }
            return;
        }
    }

    private static int LongestSuffixThatStartsMarker(string text, string marker)
    {
        var maximumLength = Math.Min(text.Length, marker.Length - 1);
        for (var length = maximumLength; length > 0; length--)
        {
            if (text.AsSpan(text.Length - length).SequenceEqual(marker.AsSpan(0, length)))
            {
                return length;
            }
        }
        return 0;
    }
}
