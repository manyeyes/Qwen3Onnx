using System.Collections.ObjectModel;
using System.Text;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Text.RegularExpressions;

namespace Qwen3.Desktop.ViewModels;

public partial class MessageViewModel : ObservableObject
{
    private string _streamingThinkingContent = string.Empty;
    private string _streamingAnswer = string.Empty;

    [ObservableProperty]
    private string role = "assistant";

    [ObservableProperty]
    private string content = string.Empty;

    [ObservableProperty]
    private string attachmentsText = string.Empty;

    [ObservableProperty]
    private ObservableCollection<string> attachmentNames = [];

    [ObservableProperty]
    private bool isPending;

    [ObservableProperty]
    private bool isThinkingExpanded;

    [ObservableProperty]
    private string promptContent = string.Empty;

    [ObservableProperty]
    private bool canRegenerate;

    [ObservableProperty]
    private bool wasInterrupted;

    [ObservableProperty]
    private bool isTextSelectionMode;

    public string InterruptionLabel => "已停止 · 部分回复已保存";
    public int BubbleColumn => IsUser ? 1 : 0;
    public int BubbleColumnSpan => IsUser ? 1 : 3;
    public double BubbleMaxWidth => IsUser ? 880 : double.PositiveInfinity;
    public Avalonia.Layout.HorizontalAlignment BubbleAlignment =>
        IsUser ? Avalonia.Layout.HorizontalAlignment.Right : Avalonia.Layout.HorizontalAlignment.Stretch;
    public Avalonia.Thickness BubbleMargin => IsUser
        ? new Avalonia.Thickness(64, 0, 8, 0)
        : new Avalonia.Thickness(8, 0, 8, 0);
    public string SelectionModeTooltip => IsTextSelectionMode ? "返回 Markdown 显示" : "切换为可选择文本";

    public string MarkdownContent
    {
        get => IsPending ? _streamingAnswer : ExtractAnswer(Content);
    }

    public string StreamingAnswer => _streamingAnswer;

    partial void OnAttachmentNamesChanged(ObservableCollection<string> value) =>
        AttachmentsText = string.Join(", ", value);

    public string ThinkingContent
    {
        get => IsPending ? _streamingThinkingContent : ExtractThinking(Content);
    }

    public bool HasThinking => ThinkingContent.Length > 0;
    public bool HasAttachments => !string.IsNullOrWhiteSpace(AttachmentsText);
    public bool IsAssistant => !IsUser;
    public bool IsNotPending => !IsPending;
    public bool IsMarkdownVisible => IsNotPending && IsAssistant && !IsTextSelectionMode;
    public string DisplayRole => IsUser ? "用户" : "助手";
    public string SelectionModeGlyph => IsTextSelectionMode ? "¶" : "▤";

    public bool IsUser => Role == "user";

    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void ToggleThinking() => IsThinkingExpanded = !IsThinkingExpanded;

    [CommunityToolkit.Mvvm.Input.RelayCommand]
    private void ToggleTextSelectionMode() => IsTextSelectionMode = !IsTextSelectionMode;

    partial void OnIsTextSelectionModeChanged(bool value) =>
        OnTextSelectionModeChanged();

    private void OnTextSelectionModeChanged()
    {
        OnPropertyChanged(nameof(SelectionModeTooltip));
        OnPropertyChanged(nameof(SelectionModeGlyph));
        OnPropertyChanged(nameof(IsMarkdownVisible));
    }

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsUser));
        OnPropertyChanged(nameof(IsAssistant));
        OnPropertyChanged(nameof(DisplayRole));
        OnPropertyChanged(nameof(BubbleColumn));
        OnPropertyChanged(nameof(BubbleAlignment));
        OnPropertyChanged(nameof(BubbleMargin));
        OnPropertyChanged(nameof(IsMarkdownVisible));
        OnPropertyChanged(nameof(BubbleMaxWidth));
        OnPropertyChanged(nameof(BubbleColumnSpan));
    }

    partial void OnContentChanged(string value)
    {
        if (!IsPending)
        {
            OnPropertyChanged(nameof(MarkdownContent));
            OnPropertyChanged(nameof(IsMarkdownVisible));
        }
    }

    public void ApplyStreamingSnapshot(string thinkingContent, string answer)
    {
        var hadThinking = _streamingThinkingContent.Length > 0;
        var thinkingChanged = !string.Equals(
            _streamingThinkingContent, thinkingContent, StringComparison.Ordinal);
        var answerChanged = !string.Equals(_streamingAnswer, answer, StringComparison.Ordinal);
        _streamingThinkingContent = thinkingContent;
        _streamingAnswer = answer;

        if (thinkingChanged)
        {
            OnPropertyChanged(nameof(ThinkingContent));
        }
        if (answerChanged)
        {
            OnPropertyChanged(nameof(StreamingAnswer));
        }
        if (hadThinking != (_streamingThinkingContent.Length > 0))
        {
            OnPropertyChanged(nameof(HasThinking));
        }
    }

    public static (string Thinking, string Answer) ParseStreamingContent(string content)
    {
        var normalized = NormalizeThinkTags(content);
        var start = normalized.IndexOf("<think>", StringComparison.Ordinal);
        var thinking = string.Empty;
        if (start >= 0)
        {
            start += "<think>".Length;
            var end = normalized.IndexOf("</think>", start, StringComparison.Ordinal);
            thinking = (end < 0 ? normalized[start..] : normalized[start..end]).Trim();
        }

        return (thinking, ExtractAnswerNormalized(normalized));
    }

    public sealed class StreamingContentParser
    {
        private static readonly string[] OpenMarkers = ["<think>", "<|think|>"];
        private static readonly string[] CloseMarkers = ["</think>", "<|/think|>"];
        private static readonly string[] AllMarkers = [.. OpenMarkers, .. CloseMarkers];
        private readonly StringBuilder _pending = new();
        private readonly StringBuilder _thinking = new();
        private readonly StringBuilder _answer = new();
        private bool _insideThinking;

        public (string Thinking, string Answer) Append(string chunk)
        {
            _pending.Append(chunk);
            while (_pending.Length > 0)
            {
                var marker = FindNextMarker();
                if (marker.Index >= 0)
                {
                    AppendText(_pending.ToString(0, marker.Index));
                    _pending.Remove(0, marker.Index + marker.Value.Length);
                    _insideThinking = marker.IsOpen;
                    continue;
                }

                var keep = GetPossibleMarkerSuffixLength();
                var emitLength = _pending.Length - keep;
                if (emitLength > 0)
                {
                    AppendText(_pending.ToString(0, emitLength));
                    _pending.Remove(0, emitLength);
                }
                break;
            }

            return (_thinking.ToString().Trim(), _answer.ToString().Trim());
        }

        private (int Index, string Value, bool IsOpen) FindNextMarker()
        {
            var pending = _pending.ToString();
            var next = (Index: -1, Value: string.Empty, IsOpen: false);
            foreach (var marker in OpenMarkers)
            {
                var index = pending.IndexOf(marker, StringComparison.Ordinal);
                if (index >= 0 && (next.Index < 0 || index < next.Index))
                {
                    next = (index, marker, true);
                }
            }
            foreach (var marker in CloseMarkers)
            {
                var index = pending.IndexOf(marker, StringComparison.Ordinal);
                if (index >= 0 && (next.Index < 0 || index < next.Index))
                {
                    next = (index, marker, false);
                }
            }
            return next;
        }

        private int GetPossibleMarkerSuffixLength()
        {
            var maxLength = Math.Min(_pending.Length, 10);
            for (var length = maxLength; length > 0; length--)
            {
                var suffix = _pending.ToString(_pending.Length - length, length);
                if (AllMarkers.Any(marker => marker.StartsWith(suffix, StringComparison.Ordinal)))
                {
                    return length;
                }
            }
            return 0;
        }

        private void AppendText(string text)
        {
            (_insideThinking ? _thinking : _answer).Append(text);
        }
    }

    partial void OnIsPendingChanged(bool value)
    {
        OnPropertyChanged(nameof(IsNotPending));
        OnPropertyChanged(nameof(IsMarkdownVisible));
        OnPropertyChanged(value ? nameof(StreamingAnswer) : nameof(MarkdownContent));
    }

    partial void OnIsThinkingExpandedChanged(bool value)
    {
        OnPropertyChanged(nameof(ThinkingMaxHeight));
        OnPropertyChanged(nameof(ThinkingToggleText));
    }

    public double ThinkingMaxHeight => IsThinkingExpanded ? 240 : 112;
    public string ThinkingToggleText => IsThinkingExpanded ? "收起思考" : "展开思考";

    private static string NormalizeThinkTags(string text) =>
        Regex.Replace(text.Replace("<|think|>", "<think>", StringComparison.Ordinal)
            .Replace("<|/think|>", "</think>", StringComparison.Ordinal),
            @"</?think[^>]*>", match => match.Value.StartsWith("</", StringComparison.Ordinal)
                ? "</think>"
                : "<think>", RegexOptions.CultureInvariant);

    private static string ExtractThinking(string content)
    {
        var normalized = NormalizeThinkTags(content);
        var start = normalized.IndexOf("<think>", StringComparison.Ordinal);
        if (start < 0)
        {
            return string.Empty;
        }

        start += "<think>".Length;
        var end = normalized.IndexOf("</think>", start, StringComparison.Ordinal);
        return (end < 0 ? normalized[start..] : normalized[start..end]).Trim();
    }

    private static string ExtractAnswer(string content)
    {
        return ExtractAnswerNormalized(NormalizeThinkTags(content));
    }

    private static string ExtractAnswerNormalized(string normalized)
    {
        var closingIndex = normalized.LastIndexOf("</think>", StringComparison.Ordinal);
        if (closingIndex >= 0)
        {
            normalized = normalized[(closingIndex + "</think>".Length)..];
        }
        else
        {
            var openingIndex = normalized.IndexOf("<think>", StringComparison.Ordinal);
            if (openingIndex >= 0)
            {
                normalized = normalized[..openingIndex];
            }
        }
        return Regex.Replace(normalized, @"<\|im_(?:start|end)\|>", string.Empty).Trim();
    }
}
