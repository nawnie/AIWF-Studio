// Precompiled header for AIWF Studio for Windows (C++/WinRT + WinUI 3).
#pragma once

// --- Windows and C++/WinRT base -----------------------------------------------------------------
#include <windows.h>
#include <unknwn.h>
#include <restrictederrorinfo.h>
#include <shobjidl_core.h>   // IInitializeWithWindow, so system pickers know their owner window
#include <hstring.h>
// performance counters: per-process GPU memory (Services/GpuMonitor)
#include <pdh.h>
#include <pdhmsg.h>

// Storyboard::GetCurrentTime collides with the GetCurrentTime macro from windows.h
#undef GetCurrentTime

// WIL's C++/WinRT interop must come before the other WIL headers and the projections
#include <wil/cppwinrt.h>
#include <wil/resource.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.ApplicationModel.Activation.h>
#include <winrt/Windows.ApplicationModel.DataTransfer.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Pickers.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.System.Threading.h>
#include <winrt/Windows.Web.Http.h>
#include <winrt/Windows.Web.Http.Headers.h>
#include <winrt/Windows.Web.Http.Filters.h>
#include <winrt/Windows.UI.h>
#include <winrt/Windows.UI.Text.h>
// xaml_typename<T>() (used to navigate a Frame to a page type) lives in this projection header,
// even in WinUI 3: page type names are still Windows.UI.Xaml.Interop.TypeName values
#include <winrt/Windows.UI.Xaml.Interop.h>

// --- WinUI 3 -----------------------------------------------------------------------------------------
#include <winrt/Microsoft.UI.h>
#include <winrt/Microsoft.UI.Composition.h>
#include <winrt/Microsoft.UI.Composition.SystemBackdrops.h>
#include <winrt/Microsoft.UI.Content.h>
#include <winrt/Microsoft.UI.Dispatching.h>
#include <winrt/Microsoft.UI.Input.h>
#include <winrt/Microsoft.UI.Windowing.h>
#include <winrt/Microsoft.UI.Interop.h>
#include <winrt/Microsoft.UI.Text.h>
#include <winrt/Microsoft.UI.Xaml.h>
#include <winrt/Microsoft.UI.Xaml.Automation.h>
#include <winrt/Microsoft.UI.Xaml.Controls.h>
#include <winrt/Microsoft.UI.Xaml.Controls.Primitives.h>
#include <winrt/Microsoft.UI.Xaml.Data.h>
#include <winrt/Microsoft.UI.Xaml.Documents.h>
#include <winrt/Microsoft.UI.Xaml.Input.h>
#include <winrt/Microsoft.UI.Xaml.Interop.h>
#include <winrt/Microsoft.UI.Xaml.Markup.h>
#include <winrt/Microsoft.UI.Xaml.Media.h>
#include <winrt/Microsoft.UI.Xaml.Media.Imaging.h>
#include <winrt/Microsoft.UI.Xaml.Navigation.h>
#include <winrt/Microsoft.UI.Xaml.Shapes.h>

#include <wil/cppwinrt_helpers.h>

// --- C++ standard library --------------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
