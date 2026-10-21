using System.Collections.ObjectModel;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using Qwen3.Desktop.Models;
using Qwen3.Desktop.Services;

namespace Qwen3.Desktop.ViewModels;

public partial class ModelsPageViewModel : ObservableObject
{
    private readonly DesktopServices _services;
    private readonly MainWindowViewModel _shell;
    private readonly ILogger<ModelsPageViewModel> _logger;

    [ObservableProperty]
    private ObservableCollection<ModelDefinition> models = [];

    [ObservableProperty]
    private string newTitle = string.Empty;

    [ObservableProperty]
    private string newEnglishName = string.Empty;

    [ObservableProperty]
    private string newDescription = string.Empty;

    [ObservableProperty]
    private string newRepositoryUrl = string.Empty;

    [ObservableProperty]
    private string newHttpFileLinks = string.Empty;

    [ObservableProperty]
    private string? editingModelId;

    [ObservableProperty]
    private bool isEditorOpen;

    [ObservableProperty]
    private bool isDeleteConfirmationOpen;

    [ObservableProperty]
    private ModelDefinition? modelPendingDelete;

    [ObservableProperty]
    private bool isBusy;

    public bool CanDownload => !IsBusy;
    public bool IsNotBusy => !IsBusy;

    [ObservableProperty]
    private double downloadProgress;

    [ObservableProperty]
    private bool isDownloadIndeterminate = true;

    [ObservableProperty]
    private string statusText = "模型文件仅在用户点击下载后获取。";

    public ModelsPageViewModel(DesktopServices services, MainWindowViewModel shell)
    {
        _services = services;
        _shell = shell;
        _logger = services.LoggerFactory.CreateLogger<ModelsPageViewModel>();
    }

    [RelayCommand]
    private void EditModel(ModelDefinition? model)
    {
        if (!IsAdmin || IsBusy || model is null)
        {
            return;
        }
        EditingModelId = model.Id;
        NewTitle = model.Title;
        NewEnglishName = model.EnglishName;
        NewDescription = model.Description;
        NewRepositoryUrl = model.RepositoryUrl;
        NewHttpFileLinks = string.Join(Environment.NewLine, model.HttpFileLinks.Select(file =>
            $"{file.Path} | {file.Url} | {file.Size} | {file.Sha256}"));
        StatusText = $"正在修改 {model.Title}。英文名称在模型安装或被对话引用后不可更改。";
        IsEditorOpen = true;
    }

    [RelayCommand]
    private void CancelEdit()
    {
        ClearEditor();
        IsEditorOpen = false;
    }

    [RelayCommand]
    private void CloseEditor()
    {
        if (!IsBusy)
        {
            CancelEdit();
        }
    }

    partial void OnIsBusyChanged(bool value)
    {
        OnPropertyChanged(nameof(CanDownload));
        OnPropertyChanged(nameof(IsNotBusy));
    }
    partial void OnEditingModelIdChanged(string? value)
    {
        OnPropertyChanged(nameof(IsEditing));
        OnPropertyChanged(nameof(EditorTitle));
    }

    public bool IsAdmin => _shell.IsAdmin;
    public bool IsEditing => EditingModelId is not null;
    public string EditorTitle => IsEditing ? "编辑模型信息" : "添加模型信息";
    public bool HasModels => Models.Count > 0;
    public bool HasNoModels => !HasModels;

    public void NotifyAdminStatusChanged() => OnPropertyChanged(nameof(IsAdmin));

    public async Task RefreshAsync(CancellationToken cancellationToken = default)
    {
        var items = await _services.Database.GetModelsAsync(cancellationToken);
        var checkedItems = items.Select(model => model with
        {
            IsInstalled = ModelDownloadService.IsInstalled(model)
        }).ToArray();
        foreach (var model in checkedItems)
        {
            if (model.IsInstalled)
            {
                var embedding = model.RequiredFiles.FirstOrDefault(path =>
                    Path.GetFileName(path).StartsWith("embedding", StringComparison.OrdinalIgnoreCase) &&
                    path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase));
                var decoder = model.RequiredFiles.FirstOrDefault(path =>
                    Path.GetFileName(path).StartsWith("decoder", StringComparison.OrdinalIgnoreCase) &&
                    path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase));
                if (embedding is not null && decoder is not null &&
                    (model.EmbeddingFileName != Path.GetFileName(embedding) ||
                     model.DecoderFileName != Path.GetFileName(decoder)))
                {
                    await _services.Database.SetModelInstalledAsync(
                        model.Id, Path.GetFileName(embedding), Path.GetFileName(decoder),
                        model.RequiredFiles, installed: true, cancellationToken);
                }
            }
        }
        Models = new ObservableCollection<ModelDefinition>(checkedItems);
        OnPropertyChanged(nameof(HasModels));
        OnPropertyChanged(nameof(HasNoModels));
    }

    [RelayCommand]
    private void OpenNewModelEditor()
    {
        if (!IsAdmin || IsBusy)
        {
            return;
        }
        ClearEditor();
        IsEditorOpen = true;
    }

    [RelayCommand]
    private async Task RegisterModelAsync(CancellationToken cancellationToken)
    {
        if (!IsAdmin || IsBusy)
        {
            StatusText = IsBusy ? "模型操作期间不能修改模型目录。" : "只有管理员可以添加模型仓库。";
            return;
        }
        IsBusy = true;
        try
        {
            var fileLinks = ParseHttpFileLinks(NewHttpFileLinks);
            if (EditingModelId is { } modelId)
            {
                await _services.Database.UpdateModelAsync(
                    modelId, NewTitle, NewEnglishName, NewDescription, NewRepositoryUrl,
                    fileLinks, cancellationToken);
                StatusText = "模型信息已更新。";
            }
            else
            {
                await _services.Database.RegisterModelAsync(
                    NewTitle, NewEnglishName, NewDescription, NewRepositoryUrl,
                    fileLinks, cancellationToken);
                StatusText = "模型信息已保存到本地目录。";
            }
            ClearEditor();
            IsEditorOpen = false;
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Model registration failed");
            StatusText = $"添加模型失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task DownloadModelAsync(ModelDefinition? model, CancellationToken cancellationToken)
    {
        if (model is null || IsBusy)
        {
            return;
        }
        IsBusy = true;
        DownloadProgress = 0;
        IsDownloadIndeterminate = true;
        try
        {
            var progress = new Progress<ModelDownloadProgress>(update =>
                Dispatcher.UIThread.Post(() =>
                {
                    StatusText = update.Message;
                    IsDownloadIndeterminate = update.Percentage is null;
                    if (update.Percentage is { } percentage)
                    {
                        DownloadProgress = Math.Clamp(percentage, 0, 100);
                    }
                }));
            var result = await _services.ModelDownloader.DownloadAsync(model, progress, cancellationToken);
            var completed = result.Model;
            await _services.Database.SetModelInstalledAsync(
                model.Id, completed.EmbeddingFileName, completed.DecoderFileName,
                completed.RequiredFiles, installed: true, cancellationToken);
            StatusText = result.CleanupWarning is null
                ? $"模型安装完成：{model.Title}"
                : $"模型安装完成：{model.Title}。{result.CleanupWarning}";
            DownloadProgress = 100;
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Model download failed for {ModelId}", model.Id);
            StatusText = $"下载失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void DeleteModel(ModelDefinition? model)
    {
        if (!IsAdmin)
        {
            StatusText = "只有管理员可以删除模型。";
            return;
        }
        if (model is null || IsBusy)
        {
            return;
        }
        ModelPendingDelete = model;
        IsDeleteConfirmationOpen = true;
    }

    [RelayCommand]
    private async Task ConfirmDeleteModelAsync(CancellationToken cancellationToken)
    {
        var model = ModelPendingDelete;
        if (!IsAdmin || model is null || IsBusy)
        {
            return;
        }
        IsBusy = true;
        try
        {
            var hasConversations = await _services.Database.HasConversationsForModelAsync(
                model.Id, cancellationToken);
            if (hasConversations)
            {
                throw new InvalidOperationException("此模型仍被历史对话引用。请先删除这些对话，再删除模型。");
            }
            if (Directory.Exists(model.LocalPath))
            {
                Directory.Delete(model.LocalPath, recursive: true);
            }
            await _services.Database.DeleteModelAsync(model.Id, cancellationToken);
            ModelPendingDelete = null;
            IsDeleteConfirmationOpen = false;
            StatusText = $"已删除模型：{model.Title}";
            await RefreshAsync(cancellationToken);
        }
        catch (Exception exception)
        {
            _logger.LogError(exception, "Model deletion failed for {ModelId}", model.Id);
            StatusText = $"删除失败：{exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void CancelDelete()
    {
        ModelPendingDelete = null;
        IsDeleteConfirmationOpen = false;
    }

    private void ClearEditor()
    {
        EditingModelId = null;
        NewTitle = string.Empty;
        NewEnglishName = string.Empty;
        NewDescription = string.Empty;
        NewRepositoryUrl = string.Empty;
        NewHttpFileLinks = string.Empty;
    }

    private static IReadOnlyList<ModelFileLink> ParseHttpFileLinks(string input)
    {
        var links = new List<ModelFileLink>();
        foreach (var (rawLine, lineIndex) in input.Split(
                     ['\r', '\n'], StringSplitOptions.RemoveEmptyEntries).Select((line, index) => (line, index)))
        {
            var parts = rawLine.Split('|').Select(part => part.Trim()).ToArray();
            if (parts.Length is < 2 or > 4 || parts[0].Length == 0 || parts[1].Length == 0)
            {
                throw new FormatException(
                    $"第 {lineIndex + 1} 行格式错误。请使用：相对文件路径 | 下载 URL | 可选大小 | 可选 SHA-256。");
            }
            var size = 0L;
            if (parts.Length >= 3 && parts[2].Length > 0 &&
                (!long.TryParse(parts[2], out size) || size < 0))
            {
                throw new FormatException($"第 {lineIndex + 1} 行文件大小必须是非负整数。");
            }
            var hash = parts.Length >= 4 && parts[3].Length > 0 ? parts[3] : null;
            links.Add(new ModelFileLink(parts[0], parts[1], size, hash));
        }
        return links;
    }
}
