using Avalonia.Controls;

namespace Qwen3.Desktop.Views;

public sealed partial class ModelsView : UserControl
{
    public ModelsView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
