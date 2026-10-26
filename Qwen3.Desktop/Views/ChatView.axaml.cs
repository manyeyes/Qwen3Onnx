using Avalonia.Controls;
using Avalonia.Threading;
using Qwen3.Desktop.ViewModels;

namespace Qwen3.Desktop.Views;

public sealed partial class ChatView : UserControl
{
    // 是否自动跟随到底部。用户手动上滚后置 false，滚回底部或点按钮后置 true。
    private bool _autoScroll = true;
    // 防止程序化滚动时递归触发 ScrollChanged
    private bool _suppressScrollChanged;
    // 消息集合订阅，用于"新增消息"时强制滚到底
    private ChatPageViewModel? _subscribedViewModel;

    public ChatView()
    {
        Avalonia.Markup.Xaml.AvaloniaXamlLoader.Load(this);
        DataContextChanged += OnDataContextChanged;
        // 布局完成后先滚一次到底部
        AttachedToVisualTree += (_, _) => Dispatcher.UIThread.Post(ScrollToBottomIfAuto, DispatcherPriority.Loaded);
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.Messages.CollectionChanged -= OnMessagesCollectionChanged;
            _subscribedViewModel.PropertyChanged -= OnViewModelPropertyChanged;
        }

        _subscribedViewModel = DataContext as ChatPageViewModel;

        if (_subscribedViewModel is not null)
        {
            _subscribedViewModel.Messages.CollectionChanged += OnMessagesCollectionChanged;
            _subscribedViewModel.PropertyChanged += OnViewModelPropertyChanged;
        }
    }

    private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
    {
        // 会话切换 / 重新加载消息时，强制跟随到底
        if (e.PropertyName == nameof(ChatPageViewModel.Messages))
        {
            if (_subscribedViewModel is not null)
            {
                _subscribedViewModel.Messages.CollectionChanged -= OnMessagesCollectionChanged;
                _subscribedViewModel.Messages.CollectionChanged += OnMessagesCollectionChanged;
            }
            _autoScroll = true;
            Dispatcher.UIThread.Post(ScrollToBottom, DispatcherPriority.Loaded);
        }
    }

    private void OnMessagesCollectionChanged(
        object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
    {
        // 新增消息（发送或收到回复）时，强制滚到底并恢复自动跟随
        if (e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Add)
        {
            _autoScroll = true;
            Dispatcher.UIThread.Post(ScrollToBottom, DispatcherPriority.Loaded);
        }
    }

    private void MessageScrollViewer_OnScrollChanged(object? sender, ScrollChangedEventArgs e)
    {
        if (_suppressScrollChanged || sender is not ScrollViewer scrollViewer)
        {
            return;
        }

        UpdateScrollState(scrollViewer);

        // 流式输出时，如果处于自动跟随状态，持续滚到底
        if (_autoScroll)
        {
            ScrollToBottom();
        }
    }

    private void ScrollToBottomButton_OnClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        _autoScroll = true;
        ScrollToBottom();
    }

    private void ScrollToBottomIfAuto()
    {
        if (_autoScroll)
        {
            ScrollToBottom();
        }
    }

    private void ScrollToBottom()
    {
        if (this.FindControl<ScrollViewer>("MessageScrollViewer") is not { } scrollViewer)
        {
            return;
        }

        _suppressScrollChanged = true;
        try
        {
            scrollViewer.ScrollToEnd();
        }
        finally
        {
            // 程序化滚动完成后，主动刷新一次状态（含按钮可见性），
            // 因为 ScrollChanged 已被抑制，不能依赖它来更新。
            Dispatcher.UIThread.Post(
                () =>
                {
                    _suppressScrollChanged = false;
                    UpdateScrollState(scrollViewer);
                },
                DispatcherPriority.Background);
        }
    }

    private void UpdateScrollState(ScrollViewer scrollViewer)
    {
        var scrollableHeight = scrollViewer.Extent.Height - scrollViewer.Viewport.Height;
        var distanceFromBottom = scrollableHeight - scrollViewer.Offset.Y;

        // 距底部小于 24px 视为"在底部"
        var atBottom = distanceFromBottom <= 24;

        if (atBottom)
        {
            // 用户滚回底部 → 恢复自动跟随
            _autoScroll = true;
        }
        else
        {
            // 用户手动上滚 → 停止跟随
            _autoScroll = false;
        }

        //if (this.FindControl<Button>("ScrollToBottomButton") is { } button)
        //{
        //    button.IsVisible = !atBottom;
        //}
        // 改为控制 Border 的可见性：
        if (this.FindControl<Border>("ScrollToBottomBorder") is { } border)
        {
            border.IsVisible = !atBottom;
        }
    }

    private void ThinkingScrollViewer_OnScrollChanged(object? sender, ScrollChangedEventArgs eventArgs)
    {
        if (sender is ScrollViewer { DataContext: MessageViewModel { IsPending: true } } scrollViewer)
        {
            scrollViewer.ScrollToEnd();
        }
    }
}