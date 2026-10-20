namespace Qwen3.Inference;

internal sealed class ConversationMemory
{
    private readonly List<ChatTurn> _turns = [];

    public int Count => _turns.Count;
    public IReadOnlyList<ChatTurn> Turns => _turns;

    public void AddTurn(string user, string assistant, int maxTurns, int maxHistoryChars)
    {
        if (assistant.Length > maxHistoryChars)
        {
            assistant = assistant[..maxHistoryChars] + "...";
        }

        _turns.Add(new ChatTurn(user, assistant));
        while (_turns.Count > maxTurns)
        {
            _turns.RemoveAt(0);
        }
    }

    public void Clear() => _turns.Clear();

    public IReadOnlyList<string> RecentAnswers(int count = 3) =>
        _turns.Skip(Math.Max(0, _turns.Count - count - 1)).Take(Math.Max(0, count)).Select(turn => turn.Assistant).ToArray();

    public void RemoveOldestTurn() => _turns.RemoveAt(0);
}
