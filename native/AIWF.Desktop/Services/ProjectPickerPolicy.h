#pragma once

namespace aiwf::ui_policy
{
    enum class ProjectPickerAction
    {
        DeferUntilClosed,
        ClearSelection,
        RefreshUnknownProject,
        KeepSelection,
        SelectKnownProject,
    };

    constexpr ProjectPickerAction DecideProjectPickerAction(bool dropDownOpen, bool hasCurrentProject,
                                                             bool currentProjectKnown, bool selectionMatches) noexcept
    {
        if (dropDownOpen) return ProjectPickerAction::DeferUntilClosed;
        if (!hasCurrentProject) return ProjectPickerAction::ClearSelection;
        if (!currentProjectKnown) return ProjectPickerAction::RefreshUnknownProject;
        return selectionMatches ? ProjectPickerAction::KeepSelection : ProjectPickerAction::SelectKnownProject;
    }

    constexpr bool NeedsProjectRefreshAfterSync(ProjectPickerAction action, bool refreshAlreadyPending,
                                                bool projectsLoaded) noexcept
    {
        if (action == ProjectPickerAction::RefreshUnknownProject) return true;
        if (action == ProjectPickerAction::ClearSelection) return refreshAlreadyPending || !projectsLoaded;
        return refreshAlreadyPending;
    }
}
