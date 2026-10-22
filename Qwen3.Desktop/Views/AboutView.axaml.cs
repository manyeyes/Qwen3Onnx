using Avalonia.Controls;

namespace Qwen3.Desktop.Views;

public sealed partial class AboutView : UserControl
{
    public AboutView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
