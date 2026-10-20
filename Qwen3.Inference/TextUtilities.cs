using System.Text.RegularExpressions;

namespace Qwen3.Inference;

internal static partial class TextUtilities
{
    private static readonly string[] RoleplayPatterns =
    [
        "你现在是", "你是一个", "你是一只", "请你扮演", "请你当", "假装你是", "你的身份是"
    ];
    private static readonly string[] PrecisePatterns = ["全文", "背诵", "原文", "一字不差", "完整地背"];
    private static readonly string[] MathPatterns = ["计算", "求解", "证明", @"求.*值", @"\d+\s*[\+\-\*/]\s*\d+"];
    private static readonly string[] FactPatterns = ["是什么$", "是哪[个首]", "谁是", "何时", "多少", "哪座", "哪首"];
    private static readonly string[] FactQuestionPatterns =
        ["是什么$", "是哪[个首]", "谁是", "何时", "多少", "全文$", "作者是", "出自哪", "哪首"];
    private static readonly string[] PronounPatterns = ["这[首个篇段]", "那[首个篇段]", "它", "他", "她"];

    public static string CleanInput(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }

        var characters = new List<char>(text.Length);
        foreach (var character in text)
        {
            if (character is '\b' or '\x7f')
            {
                if (characters.Count > 0)
                {
                    characters.RemoveAt(characters.Count - 1);
                }
            }
            else
            {
                characters.Add(character);
            }
        }

        var withoutEscapeSequences = EscapeSequenceRegex().Replace(
            new string(characters.ToArray()), string.Empty);
        return new string(withoutEscapeSequences.Where(character => character == '\t' || character >= ' ').ToArray()).Trim();
    }

    public static Scene DetectScene(string question)
    {
        if (question.Length == 0)
        {
            return Scene.Chitchat;
        }

        if (MatchesAny(RoleplayPatterns, question))
        {
            return Scene.Roleplay;
        }
        if (MatchesAny(PrecisePatterns, question))
        {
            return Scene.Precise;
        }
        if (MatchesAny(MathPatterns, question))
        {
            return Scene.DeepReasoning;
        }
        if (question.Length <= 15 && MatchesAny(FactPatterns, question))
        {
            return Scene.FastQa;
        }

        return Scene.Chitchat;
    }

    public static string NormalizeThinkTags(string text)
    {
        text = Regex.Replace(text, @"<think[^>]*>", "<think>", RegexOptions.CultureInvariant);
        text = Regex.Replace(text, @"</think[^>]*>", "</think>", RegexOptions.CultureInvariant);
        text = text.Replace("<|think|>", "<think>", StringComparison.Ordinal)
            .Replace("<|/think|>", "</think>", StringComparison.Ordinal);
        return text;
    }

    public static string ExtractAnswerForHistory(string text)
    {
        text = NormalizeThinkTags(text);
        var endThinkIndex = text.LastIndexOf("</think>", StringComparison.Ordinal);
        if (endThinkIndex >= 0)
        {
            text = text[(endThinkIndex + "</think>".Length)..];
        }
        else
        {
            var thinkIndex = text.IndexOf("<think>", StringComparison.Ordinal);
            if (thinkIndex >= 0)
            {
                text = text[(thinkIndex + "<think>".Length)..];
            }
        }

        text = Regex.Replace(text, @"<think\s*>|</think\s*>|<\|im_start\|>|<\|im_end\|>", string.Empty);
        return Regex.Replace(text.Trim(), @"^[\s:：,，.。;；、>]+", string.Empty).Trim();
    }

    public static string SanitizeRoleplay(string text)
    {
        text = Regex.Replace(text, @"^[（(][^）)]*[）)]\s*", string.Empty);
        text = Regex.Replace(text, @"[（(][^）)]{0,60}[）)]", string.Empty);
        return Regex.Replace(text, @"\s{2,}", " ").Trim();
    }

    public static bool IsFactQuestion(string question)
    {
        if (MatchesAny(PronounPatterns, question))
        {
            return false;
        }

        return MatchesAny(FactQuestionPatterns, question);
    }

    public static bool IsRepetition(string answer, IReadOnlyList<string> recentAnswers, double threshold)
    {
        var answerHead = answer.Length > 80 ? answer[..80].Trim() : answer.Trim();
        if (answerHead.Length < 20)
        {
            return false;
        }

        foreach (var previous in recentAnswers.TakeLast(3))
        {
            var previousHead = previous.Length > 80 ? previous[..80].Trim() : previous.Trim();
            if (previousHead.Length < 20 ||
                (double)Math.Min(answerHead.Length, previousHead.Length) /
                Math.Max(answerHead.Length, previousHead.Length) < 0.7)
            {
                continue;
            }

            if (Similarity(answerHead, previousHead) > threshold)
            {
                return true;
            }
        }

        return false;
    }

    private static bool MatchesAny(IEnumerable<string> patterns, string value) =>
        patterns.Any(pattern => Regex.IsMatch(value, pattern, RegexOptions.CultureInvariant));

    private static double Similarity(string left, string right)
    {
        var previous = Enumerable.Range(0, right.Length + 1).ToArray();
        var current = new int[right.Length + 1];
        for (var i = 1; i <= left.Length; i++)
        {
            current[0] = i;
            for (var j = 1; j <= right.Length; j++)
            {
                current[j] = Math.Min(
                    Math.Min(current[j - 1] + 1, previous[j] + 1),
                    previous[j - 1] + (left[i - 1] == right[j - 1] ? 0 : 1));
            }

            (previous, current) = (current, previous);
        }

        return 1.0 - (double)previous[right.Length] / Math.Max(left.Length, right.Length);
    }

    [GeneratedRegex(@"\x1b\[[0-9;]*[A-Za-z]", RegexOptions.CultureInvariant)]
    private static partial Regex EscapeSequenceRegex();
}
