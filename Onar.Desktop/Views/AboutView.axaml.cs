using Avalonia.Controls;

namespace Onar.Desktop.Views;

public sealed partial class AboutView : UserControl
{
    public AboutView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
