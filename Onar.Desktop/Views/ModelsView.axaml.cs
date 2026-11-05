using Avalonia.Controls;

namespace Onar.Desktop.Views;

public sealed partial class ModelsView : UserControl
{
    public ModelsView() => Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
}
