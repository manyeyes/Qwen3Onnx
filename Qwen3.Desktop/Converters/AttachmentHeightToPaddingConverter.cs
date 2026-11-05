using System;
using System.Globalization;
using Avalonia;
using Avalonia.Data.Converters;

namespace Qwen3.Desktop.Converters;

/// <summary>
/// 把附件区实际高度（double）转换为 TextBox 的 Padding（Thickness）。
/// 顶部 = 附件区高度 + 8（无附件时为 10）；底部固定 52 给工具栏。
/// </summary>
public sealed class AttachmentHeightToPaddingConverter : IValueConverter
{
    public static readonly AttachmentHeightToPaddingConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        var h = value is double d && d > 0 ? d : 0;
        var top = h > 0 ? h + 8 : 10;
        return new Thickness(14, top, 14, 52);
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        => throw new NotSupportedException();
}