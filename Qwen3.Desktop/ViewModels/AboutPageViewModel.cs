using System.Reflection;

namespace Qwen3.Desktop.ViewModels;

public sealed class AboutPageViewModel
{
    private static readonly Assembly ApplicationAssembly = Assembly.GetEntryAssembly()
        ?? typeof(AboutPageViewModel).Assembly;

    public string ApplicationName => "Qwen3 ONNX";
    public string Version => ApplicationAssembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
        ?.InformationalVersion
        ?? ApplicationAssembly.GetName().Version?.ToString()
        ?? "未知";
    public string RuntimeVersion => Environment.Version.ToString();
    public string Description =>
        "Qwen3 ONNX 是一款基于 .NET、ONNX Runtime 与 Avalonia 构建的跨平台本地 AI 对话应用。";
    public string ProductDetails =>
        "支持本地 Qwen3 ONNX 模型推理、思考模式、流式聊天、Markdown 消息、SQLite 历史对话、模型管理与本地账户。";
    public string Copyright => $"© {DateTime.Now.Year} manyeyes";
}
